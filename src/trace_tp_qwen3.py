"""
trace_tp_qwen3.py -- Subprocess script to trace a TP Qwen3 model to Neuron.

Supports batch_size=2 tracing for classifier-free guidance (CFG) optimization.
Both batch=1 and batch=2 traces are generated per bucket for flexibility.

Compiler optimization flags (NXD inference best practices + NKI support):
  --model-type=transformer    Transformer-specific Neuron compiler optimizations
  --auto-cast=matmult         Cast matmuls to BF16
  --auto-cast-type=bf16       BF16 target
  -O2                         Full optimization
  --enable-ccop-compute-overlap  Pipeline compute+communication overlap
  --cc-pipeline-tiling-factor=2  Optimal tiling factor for TP=2
  --vectorize-strided-dma     Vectorize strided DMA patterns
  --enable-saturate-infinity  Saturate inf values for numerical stability
  --retry_failed_compilation  Retry failed graph compilation passes

Usage:
    python trace_tp_qwen3.py <weights_path> <trace_dir> <bucket_size> \
        <num_layers> <hidden_dim> <num_heads> <num_kv_heads> \
        <intermediate_size> <tp_degree> [<rope_theta> [<head_dim> [<batch_size>]]]
"""

import sys
import os
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="torch_neuronx.nki_jit is deprecated",
    category=DeprecationWarning,
)

import types as _types
import torch

# Workaround: neuronx_distributed imports transformers.utils.fx which was
# removed in transformers>=5.0. Provide a stub so the import succeeds.
if "transformers.utils.fx" not in sys.modules:
    _fx_stub = _types.ModuleType("transformers.utils.fx")
    _fx_stub.HFTracer = type("HFTracer", (), {})
    sys.modules["transformers.utils.fx"] = _fx_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed.trace import parallel_model_trace, parallel_model_save
from tp_qwen3_model import TPQwen3Factory

# Neuron compiler flags -- same proven flags as original, now with batch=2 support
COMPILER_ARGS = (
    "--model-type=transformer "
    "--auto-cast=matmult --auto-cast-type=bf16 "
    "-O2 "
    "--tensorizer-options='"
    "--enable-ccop-compute-overlap "
    "--cc-pipeline-tiling-factor=2 "
    "--vectorize-strided-dma'"
)


def trace_bucket(factory, bucket_size, hidden_dim, tp_degree, batch_size,
                 save_dir):
    """Trace a single bucket with given batch_size."""
    os.makedirs(save_dir, exist_ok=True)

    example_inputs = (
        torch.randn(batch_size, bucket_size, hidden_dim, dtype=torch.bfloat16),
        torch.ones(batch_size, bucket_size, dtype=torch.bfloat16),
    )

    print(f"Tracing TP Qwen3 (tp={tp_degree}, bucket={bucket_size}, "
          f"batch={batch_size})...", flush=True)
    t0 = time.perf_counter()
    traced = parallel_model_trace(
        factory,
        example_inputs,
        tp_degree=tp_degree,
        compiler_args=COMPILER_ARGS,
    )
    elapsed = time.perf_counter() - t0
    print(f"Traced bucket {bucket_size} batch={batch_size} in {elapsed:.1f}s",
          flush=True)

    parallel_model_save(traced, save_dir)
    print(f"Saved to {save_dir}", flush=True)


def main():
    weights_path = sys.argv[1]
    trace_dir = sys.argv[2]
    bucket_size = int(sys.argv[3])
    num_layers = int(sys.argv[4])
    hidden_dim = int(sys.argv[5])
    num_heads = int(sys.argv[6])
    num_kv_heads = int(sys.argv[7])
    intermediate_size = int(sys.argv[8])
    tp_degree = int(sys.argv[9])
    rope_theta = float(sys.argv[10]) if len(sys.argv) > 10 else 1000000.0
    head_dim = int(sys.argv[11]) if len(sys.argv) > 11 else None
    if head_dim is not None and head_dim <= 0:
        head_dim = None
    batch_size = int(sys.argv[12]) if len(sys.argv) > 12 else 2
    audio_vocab_size = int(sys.argv[13]) if len(sys.argv) > 13 else 0
    max_position = 4096

    factory = TPQwen3Factory(
        weights_path, num_layers, hidden_dim, num_heads,
        num_kv_heads, intermediate_size, max_position, tp_degree,
        head_dim=head_dim, rope_theta=rope_theta,
        audio_vocab_size=audio_vocab_size,
    )

    # Use _fused suffix when audio_heads are fused into the backbone
    fused_suffix = "_fused" if audio_vocab_size > 0 else ""

    # Trace batch=2 (for CFG -- conditional + unconditional in single call)
    save_dir_b2 = os.path.join(
        trace_dir, f"qwen3_tp{tp_degree}_{bucket_size}{fused_suffix}_b2"
    )
    trace_bucket(factory, bucket_size, hidden_dim, tp_degree, 2, save_dir_b2)

    # Also trace batch=1 (for non-CFG or single-item inference)
    save_dir_b1 = os.path.join(
        trace_dir, f"qwen3_tp{tp_degree}_{bucket_size}{fused_suffix}"
    )
    trace_bucket(factory, bucket_size, hidden_dim, tp_degree, 1, save_dir_b1)


if __name__ == "__main__":
    main()
