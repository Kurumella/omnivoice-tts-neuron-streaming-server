"""
nki_kernels.py -- NKI (Neuron Kernel Interface) fused kernels for OmniVoice TTS.

Custom NeuronCore kernels that fuse multiple operations to minimize HBM
round-trips and maximize compute utilization on NeuronCore-v2.

Kernels:
  - nki_fused_rmsnorm:  Hardware-accelerated RMS normalization with weight scaling
  - nki_fused_swiglu:   Fused SiLU(gate) * up (eliminates intermediate HBM write)
  - nki_fused_rope:     Fused RoPE for Q and K (compute rotate_half in SBUF)

Hardware: NeuronCore-v2 (trn1/inf2)
  - Partition dim (P): max 128 -- first indexing dimension
  - Free dim (F): flexible   -- second indexing dimension

NKI SDK v2.23 pattern: output tensors are allocated inside the kernel with
nl.ndarray(..., buffer=nl.shared_hbm). Parameters are immutable inputs.
"""

import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

TILE_P = nl.tile_size.pmax  # 128


# ---------------------------------------------------------------------------
# Fused RMSNorm
# ---------------------------------------------------------------------------

@nki.jit
def nki_fused_rmsnorm_kernel(x_ref, weight_ref, hidden_dim, eps):
    """Fused RMSNorm: out = rms_norm(x) * weight  (all in SBUF).

    Layout: x_ref[batch_seq, hidden_dim], weight_ref[1, hidden_dim]
    Output allocated internally (NKI immutable-parameter pattern).
    """
    batch_seq = x_ref.shape[0]
    out_ref = nl.ndarray(x_ref.shape, dtype=x_ref.dtype, buffer=nl.shared_hbm)

    # Load weight into SBUF once -- hidden_dim in free dim (no P overflow)
    i_w_p = nl.arange(1)[:, None]
    i_w_f = nl.arange(hidden_dim)[None, :]
    w_sbuf = nl.load(weight_ref[i_w_p, i_w_f])  # [1, hidden_dim]

    for i in nl.affine_range(batch_seq // TILE_P):
        i_p = i * TILE_P + nl.arange(TILE_P)[:, None]
        i_f = nl.arange(hidden_dim)[None, :]

        x_tile = nl.load(x_ref[i_p, i_f])  # [TILE_P, hidden_dim]

        # Hardware-accelerated RMSNorm: normalize + scale in single pass
        normed = nl.rms_norm(x_tile, w_sbuf, axis=1, n=hidden_dim, epsilon=eps)

        nl.store(out_ref[i_p, i_f], normed)

    return out_ref


# ---------------------------------------------------------------------------
# Fused SwiGLU
# ---------------------------------------------------------------------------

@nki.jit
def nki_fused_swiglu_kernel(gate_ref, up_ref):
    """Fused SwiGLU: out = SiLU(gate) * up  (all in SBUF).

    Layout: gate_ref[batch_seq, local_intermediate]
    Output allocated internally.
    """
    batch_seq, local_intermediate = gate_ref.shape
    out_ref = nl.ndarray(gate_ref.shape, dtype=gate_ref.dtype, buffer=nl.shared_hbm)

    for i in nl.affine_range(batch_seq // TILE_P):
        i_p = i * TILE_P + nl.arange(TILE_P)[:, None]
        i_f = nl.arange(local_intermediate)[None, :]

        gate_tile = nl.load(gate_ref[i_p, i_f])
        up_tile = nl.load(up_ref[i_p, i_f])

        # Fused: SiLU(gate) * up -- no intermediate HBM write
        activated = nl.silu(gate_tile)
        result = nl.multiply(activated, up_tile)

        nl.store(out_ref[i_p, i_f], result)

    return out_ref


# ---------------------------------------------------------------------------
# Fused RoPE (single tensor)
# ---------------------------------------------------------------------------

@nki.jit
def nki_rope_single_kernel(x_ref, cos_ref, sin_ref, half_dim):
    """Apply RoPE to a single tensor (Q or K) entirely in SBUF.

    result_lo = x_lo * cos_lo - x_hi * sin_lo
    result_hi = x_hi * cos_hi + x_lo * sin_hi

    Layout: x_ref[total_rows, head_dim], cos/sin same shape
    Output allocated internally.
    """
    total_rows, head_dim = x_ref.shape
    out_ref = nl.ndarray(x_ref.shape, dtype=x_ref.dtype, buffer=nl.shared_hbm)

    for i in nl.affine_range(total_rows // TILE_P):
        i_p = i * TILE_P + nl.arange(TILE_P)[:, None]
        i_f_lo = nl.arange(half_dim)[None, :]
        i_f_hi = half_dim + nl.arange(half_dim)[None, :]

        # Load input halves
        x_lo = nl.load(x_ref[i_p, i_f_lo])
        x_hi = nl.load(x_ref[i_p, i_f_hi])

        # Load cos/sin halves
        cos_lo = nl.load(cos_ref[i_p, i_f_lo])
        cos_hi = nl.load(cos_ref[i_p, i_f_hi])
        sin_lo = nl.load(sin_ref[i_p, i_f_lo])
        sin_hi = nl.load(sin_ref[i_p, i_f_hi])

        # Fused rotate_half + multiply (all in SBUF, no HBM round-trip)
        out_lo = nl.subtract(nl.multiply(x_lo, cos_lo),
                             nl.multiply(x_hi, sin_lo))
        out_hi = nl.add(nl.multiply(x_hi, cos_hi),
                        nl.multiply(x_lo, sin_hi))

        nl.store(out_ref[i_p, i_f_lo], out_lo)
        nl.store(out_ref[i_p, i_f_hi], out_hi)

    return out_ref


# ---------------------------------------------------------------------------
# PyTorch wrapper: fused RMSNorm
# ---------------------------------------------------------------------------

def fused_rmsnorm(x, weight, eps=1e-6):
    """PyTorch-callable fused RMSNorm via NKI."""
    orig_shape = x.shape
    hidden_dim = x.shape[-1]
    x_2d = x.reshape(-1, hidden_dim).contiguous()
    batch_seq = x_2d.shape[0]

    # Pad to TILE_P alignment if needed
    pad_len = (TILE_P - batch_seq % TILE_P) % TILE_P
    if pad_len:
        x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_len))

    w_2d = weight.unsqueeze(0).contiguous()  # [1, hidden_dim]
    out = nki_fused_rmsnorm_kernel(x_2d, w_2d, hidden_dim, eps)

    if pad_len:
        out = out[:batch_seq]
    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# PyTorch wrapper: fused SwiGLU
# ---------------------------------------------------------------------------

def fused_swiglu(gate, up):
    """PyTorch-callable fused SwiGLU: SiLU(gate) * up via NKI."""
    orig_shape = gate.shape
    intermediate = gate.shape[-1]
    gate_2d = gate.reshape(-1, intermediate).contiguous()
    up_2d = up.reshape(-1, intermediate).contiguous()
    batch_seq = gate_2d.shape[0]

    pad_len = (TILE_P - batch_seq % TILE_P) % TILE_P
    if pad_len:
        gate_2d = torch.nn.functional.pad(gate_2d, (0, 0, 0, pad_len))
        up_2d = torch.nn.functional.pad(up_2d, (0, 0, 0, pad_len))

    out = nki_fused_swiglu_kernel(gate_2d, up_2d)

    if pad_len:
        out = out[:batch_seq]
    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# PyTorch wrapper: fused RoPE for Q and K
# ---------------------------------------------------------------------------

def _apply_rope_single(x_2d, cos_2d, sin_2d, half_dim):
    """Apply NKI RoPE kernel to a single 2D tensor with TILE_P padding."""
    total = x_2d.shape[0]
    pad = (TILE_P - total % TILE_P) % TILE_P
    if pad:
        x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad))
        cos_2d = torch.nn.functional.pad(cos_2d, (0, 0, 0, pad))
        sin_2d = torch.nn.functional.pad(sin_2d, (0, 0, 0, pad))

    out = nki_rope_single_kernel(x_2d, cos_2d, sin_2d, half_dim)

    if pad:
        out = out[:total]
    return out


def fused_rope_qk(q, k, cos, sin):
    """PyTorch-callable fused RoPE for Q and K via NKI."""
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    half_dim = head_dim // 2

    # Flatten to 2D: [batch * heads * seq, head_dim]
    q_2d = q.reshape(-1, head_dim).contiguous()
    k_2d = k.reshape(-1, head_dim).contiguous()

    # Expand cos/sin to match Q and K row counts
    cos_q = cos.expand(batch, num_q_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()
    sin_q = sin.expand(batch, num_q_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()
    cos_k = cos.expand(batch, num_kv_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()
    sin_k = sin.expand(batch, num_kv_heads, seq_len, head_dim).reshape(-1, head_dim).contiguous()

    # Apply RoPE separately for Q and K (different row counts with GQA)
    q_out = _apply_rope_single(q_2d, cos_q, sin_q, half_dim)
    k_out = _apply_rope_single(k_2d, cos_k, sin_k, half_dim)

    q_embed = q_out.reshape(batch, num_q_heads, seq_len, head_dim)
    k_embed = k_out.reshape(batch, num_kv_heads, seq_len, head_dim)
    return q_embed, k_embed
