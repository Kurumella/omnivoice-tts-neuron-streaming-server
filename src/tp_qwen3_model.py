"""
tp_qwen3_model.py -- Tensor-Parallel Qwen3 Transformer for AWS Neuron (TP=2)

Optimized with NKI fused kernels and batch=2 CFG support.

Sharding strategy per transformer block (GQA-aware):
  - q_proj:    ColumnParallel -- each core gets half the Q heads (8 of 16)
  - k_proj:    ColumnParallel -- each core gets half the KV heads (4 of 8)
  - v_proj:    ColumnParallel -- each core gets half the KV heads (4 of 8)
  - o_proj:    RowParallel    -- each core holds half, all-reduced
  - gate_proj: ColumnParallel -- each core gets half intermediate dim
  - up_proj:   ColumnParallel -- each core gets half intermediate dim
  - down_proj: RowParallel    -- each core holds half, all-reduced
  - RMSNorm:   Replicated     -- identical on both cores

NKI kernel optimizations:
  - Fused RMSNorm:  nl.rms_norm hardware primitive (no HBM round-trip)
  - Fused SwiGLU:   SiLU(gate) * up in single kernel (eliminates intermediate)
  - Fused RoPE:     Q and K rotate_half computed in SBUF

Batch=2 support:
  - Traced with batch_size=2 for classifier-free guidance (CFG)
  - Eliminates sequential per-sample Neuron calls in diffusion loop
"""

import warnings
import math

warnings.filterwarnings(
    "ignore",
    message="torch_neuronx.nki_jit is deprecated",
    category=DeprecationWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Workaround: neuronx_distributed imports transformers.utils.fx which was
# removed in transformers>=5.0. Provide a stub so the import succeeds.
import sys
import types as _types
if "transformers.utils.fx" not in sys.modules:
    _fx_stub = _types.ModuleType("transformers.utils.fx")
    _fx_stub.HFTracer = type("HFTracer", (), {})
    sys.modules["transformers.utils.fx"] = _fx_stub

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state

# NKI fused kernels -- RMSNorm, SwiGLU, RoPE computed in SBUF without HBM
# round-trips.  Weight loads use [1, hidden_dim] layout to keep hidden_dim
# in the free dimension (avoids P_max=128 overflow).
try:
    from nki_kernels import fused_rmsnorm, fused_swiglu, fused_rope_qk
    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False


# -- RMSNorm -----------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Layer Normalization with optional NKI fusion."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x):
        if _NKI_AVAILABLE:
            return fused_rmsnorm(x, self.weight, self.eps)
        # Fallback: PyTorch implementation
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


# -- Rotary Position Embeddings ----------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for Qwen3."""

    def __init__(self, dim, max_position=4096, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x):
    """Rotate half the hidden dims of x for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to Q and K tensors."""
    if _NKI_AVAILABLE:
        return fused_rope_qk(q, k, cos, sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -- Transformer block -------------------------------------------------------

class TPQwen3Block(nn.Module):
    """Single Qwen3 transformer block with TP attention/MLP and NKI fusion.

    Attention: GQA with Q=num_heads, KV=num_kv_heads
    MLP: SwiGLU (gate * up then down) with NKI fused activation
    Norm: RMSNorm with NKI hardware primitive
    """

    def __init__(self, hidden_dim, num_heads, num_kv_heads, intermediate_size,
                 tp_degree, head_dim=None, rope_theta=1000000.0, max_position=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # Local dimensions after TP sharding
        self.local_q_heads = num_heads // tp_degree
        self.local_kv_heads = num_kv_heads // tp_degree
        self.local_q_dim = self.local_q_heads * self.head_dim
        self.local_kv_dim = self.local_kv_heads * self.head_dim

        # Pre-attention norm (NKI-accelerated)
        self.input_layernorm = RMSNorm(hidden_dim)

        # GQA attention projections
        self.q_proj = ColumnParallelLinear(
            hidden_dim, num_heads * self.head_dim,
            bias=False, gather_output=False)
        self.k_proj = ColumnParallelLinear(
            hidden_dim, num_kv_heads * self.head_dim,
            bias=False, gather_output=False)
        self.v_proj = ColumnParallelLinear(
            hidden_dim, num_kv_heads * self.head_dim,
            bias=False, gather_output=False)
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim, hidden_dim,
            bias=False, input_is_parallel=True)

        # Qwen3 has per-head norms for Q and K
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Post-attention norm (NKI-accelerated)
        self.post_attention_layernorm = RMSNorm(hidden_dim)

        # SwiGLU MLP
        self.gate_proj = ColumnParallelLinear(
            hidden_dim, intermediate_size,
            bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(
            hidden_dim, intermediate_size,
            bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_dim,
            bias=False, input_is_parallel=True)

        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position, rope_theta)

    def forward(self, hidden, attention_mask=None):
        residual = hidden

        # Pre-norm (NKI fused RMSNorm)
        normed = self.input_layernorm(hidden)

        # QKV projections (sharded by TP)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        batch, seq_len = q.shape[:2]

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.local_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.local_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head RMSNorm on Q and K (Qwen3 feature)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE (NKI fused for Q and K simultaneously)
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads to match Q heads per rank
        local_kv_repeat = self.local_q_heads // self.local_kv_heads
        if local_kv_repeat > 1:
            k = k.repeat_interleave(local_kv_repeat, dim=1)
            v = v.repeat_interleave(local_kv_repeat, dim=1)

        # Bidirectional attention (is_causal=False for masked diffusion)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch, seq_len, self.local_q_dim)
        attn_out = self.o_proj(attn_out)
        hidden = residual + attn_out

        # MLP with NKI fused SwiGLU
        residual = hidden
        normed = self.post_attention_layernorm(hidden)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)

        if _NKI_AVAILABLE:
            mlp_out = fused_swiglu(gate, up)
        else:
            mlp_out = F.silu(gate) * up

        mlp_out = self.down_proj(mlp_out)
        hidden = residual + mlp_out

        return hidden


# -- Full transformer ---------------------------------------------------------

class TPQwen3Transformer(nn.Module):
    """Full Qwen3 transformer stack with TP, NKI kernels, and batch=2 support.

    Takes pre-computed embeddings as input. Supports batch_size=2 natively
    for classifier-free guidance (CFG) without sequential per-sample calls.

    When audio_vocab_size > 0, includes a fused audio projection head
    (ColumnParallel) that runs on NeuronCores, eliminating the CPU-side
    audio_heads bottleneck (~80ms/step saved).
    """

    def __init__(self, num_layers, hidden_dim, num_heads, num_kv_heads,
                 intermediate_size, max_position, tp_degree,
                 head_dim=None, rope_theta=1000000.0, audio_vocab_size=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TPQwen3Block(
                hidden_dim, num_heads, num_kv_heads, intermediate_size,
                tp_degree, head_dim=head_dim, rope_theta=rope_theta,
                max_position=max_position,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)

        # Fused audio projection head on NeuronCores (TP-sharded)
        self.audio_vocab_size = audio_vocab_size
        if audio_vocab_size > 0:
            self.audio_projection = ColumnParallelLinear(
                hidden_dim, audio_vocab_size,
                bias=False, gather_output=True,
            )
        else:
            self.audio_projection = None

    def forward(self, inputs_embeds, attention_mask=None):
        """Forward pass through the transformer.

        Supports batch_size >= 1 natively (batch=2 for CFG).

        Args:
            inputs_embeds: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) - 1=real, 0=pad

        Returns:
            If audio_projection is present:
                logits: (batch, seq_len, audio_vocab_size)
            Else:
                hidden_states: (batch, seq_len, hidden_dim)
        """
        hidden = inputs_embeds

        # Convert 2D mask to 4D for SDPA: (batch, 1, seq_len, seq_len)
        if attention_mask is not None and attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden.dtype).min
        else:
            expanded_mask = None

        for block in self.blocks:
            hidden = block(hidden, expanded_mask)

        hidden = self.norm(hidden)

        if self.audio_projection is not None:
            return self.audio_projection(hidden)
        return hidden


# -- Weight extraction --------------------------------------------------------

def extract_qwen3_weights(qwen3_model, audio_heads=None):
    """Extract Qwen3 weights from a HuggingFace Qwen3Model.

    Args:
        qwen3_model: The Qwen3 backbone (model.llm).
        audio_heads: Optional nn.Linear audio_heads to fuse into the trace.
            When provided, the traced model will output logits directly,
            eliminating the CPU audio_heads bottleneck.
    """
    layers = []
    for i in range(len(qwen3_model.layers)):
        layer = qwen3_model.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        layers.append({
            "q_proj_w": attn.q_proj.weight.data.clone(),
            "k_proj_w": attn.k_proj.weight.data.clone(),
            "v_proj_w": attn.v_proj.weight.data.clone(),
            "o_proj_w": attn.o_proj.weight.data.clone(),
            "q_norm_w": attn.q_norm.weight.data.clone(),
            "k_norm_w": attn.k_norm.weight.data.clone(),
            "input_ln_w": layer.input_layernorm.weight.data.clone(),
            "post_attn_ln_w": layer.post_attention_layernorm.weight.data.clone(),
            "gate_proj_w": mlp.gate_proj.weight.data.clone(),
            "up_proj_w": mlp.up_proj.weight.data.clone(),
            "down_proj_w": mlp.down_proj.weight.data.clone(),
        })

    result = {
        "layers": layers,
        "norm_w": qwen3_model.norm.weight.data.clone(),
    }

    if audio_heads is not None:
        result["audio_heads_w"] = audio_heads.weight.data.clone()
        if audio_heads.bias is not None:
            result["audio_heads_b"] = audio_heads.bias.data.clone()

    return result


# -- Factory for parallel_model_trace ----------------------------------------

class TPQwen3Factory:
    """Picklable factory that creates a TP Qwen3 model and loads sharded weights.

    Supports configurable batch_size for tracing (batch=2 for CFG).
    When audio_vocab_size > 0, the model includes a fused audio_heads
    projection that runs on NeuronCores (TP-sharded ColumnParallel).
    """

    def __init__(self, weights_path, num_layers, hidden_dim, num_heads,
                 num_kv_heads, intermediate_size, max_position, tp_degree,
                 head_dim=None, rope_theta=1000000.0, audio_vocab_size=0):
        self.weights_path = weights_path
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_position = max_position
        self.tp_degree = tp_degree
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.audio_vocab_size = audio_vocab_size

    def __call__(self):
        model = TPQwen3Transformer(
            self.num_layers, self.hidden_dim, self.num_heads,
            self.num_kv_heads, self.intermediate_size,
            self.max_position, self.tp_degree,
            head_dim=self.head_dim, rope_theta=self.rope_theta,
            audio_vocab_size=self.audio_vocab_size,
        )
        model = model.to(torch.bfloat16)
        model.eval()

        weights = torch.load(self.weights_path, map_location="cpu")
        rank = parallel_state.get_tensor_model_parallel_rank()
        tp = self.tp_degree

        # Final norm
        model.norm.weight.data.copy_(weights["norm_w"].to(torch.bfloat16))

        for i, block in enumerate(model.blocks):
            lw = weights["layers"][i]

            # RMSNorms (replicated)
            block.input_layernorm.weight.data.copy_(
                lw["input_ln_w"].to(torch.bfloat16))
            block.post_attention_layernorm.weight.data.copy_(
                lw["post_attn_ln_w"].to(torch.bfloat16))

            # Q projection: ColumnParallel -- shard Q heads
            w = lw["q_proj_w"].to(torch.bfloat16)
            q_total = w.shape[0]
            q_chunk = q_total // tp
            block.q_proj.weight.data.copy_(w[rank * q_chunk:(rank + 1) * q_chunk])

            # K projection: ColumnParallel -- shard KV heads
            w = lw["k_proj_w"].to(torch.bfloat16)
            kv_total = w.shape[0]
            kv_chunk = kv_total // tp
            block.k_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

            # V projection: ColumnParallel -- shard KV heads
            w = lw["v_proj_w"].to(torch.bfloat16)
            block.v_proj.weight.data.copy_(w[rank * kv_chunk:(rank + 1) * kv_chunk])

            # O projection: RowParallel -- shard input dim
            w = lw["o_proj_w"].to(torch.bfloat16)
            o_chunk = w.shape[1] // tp
            block.o_proj.weight.data.copy_(w[:, rank * o_chunk:(rank + 1) * o_chunk])

            # Per-head Q norm: replicated
            qn_w = lw["q_norm_w"].to(torch.bfloat16)
            block.q_norm.weight.data.copy_(qn_w)

            # Per-head K norm: replicated
            kn_w = lw["k_norm_w"].to(torch.bfloat16)
            block.k_norm.weight.data.copy_(kn_w)

            # Gate projection: ColumnParallel
            w = lw["gate_proj_w"].to(torch.bfloat16)
            gate_chunk = w.shape[0] // tp
            block.gate_proj.weight.data.copy_(
                w[rank * gate_chunk:(rank + 1) * gate_chunk])

            # Up projection: ColumnParallel
            w = lw["up_proj_w"].to(torch.bfloat16)
            up_chunk = w.shape[0] // tp
            block.up_proj.weight.data.copy_(
                w[rank * up_chunk:(rank + 1) * up_chunk])

            # Down projection: RowParallel
            w = lw["down_proj_w"].to(torch.bfloat16)
            down_chunk = w.shape[1] // tp
            block.down_proj.weight.data.copy_(
                w[:, rank * down_chunk:(rank + 1) * down_chunk])

        # Audio projection head: ColumnParallel (gather_output=True)
        if model.audio_projection is not None and "audio_heads_w" in weights:
            w = weights["audio_heads_w"].to(torch.bfloat16)
            out_chunk = w.shape[0] // tp
            model.audio_projection.weight.data.copy_(
                w[rank * out_chunk:(rank + 1) * out_chunk])

        return model, None
