# Projection cuBLASLt plan stack screens

These screens modify only the static H200 plan descriptor in the attributed Nathen v18 projection submission. The base source is frozen at `experiments/projection_nathen_v18/submission.py`; every variant passed all 9 official correctness tests. No candidate was publicly submitted.

## Findings

| Variant | Change | Official speedup | Target profile | Decision |
|---|---|---:|---:|---|
| `projection_pool256_alt1` | M256 pooler, layout 3 alternate | 1.526x | 25.6 us vs prior 24.2 us | Reject; target regressed |
| `projection_pool256_alt2` | M256 pooler, layout 0 alternate | 1.503x | 27.8 us vs prior 24.2 us | Reject; target regressed |
| `projection_unpool128_alt1` | M128 unpooler alternate | 1.530x | 18.1 us vs prior 19.1 us | Repeat was 1.510x; no robust aggregate gain |
| `projection_unpool128_alt1_repeat` | Same kernel; comment-only SHA change for a fresh judge run | 1.510x | 18.8 us | No aggregate gain |
| `projection_128_lt_stack` | M128 pooler + unpooler alternates | 1.524x | 20.6 / 18.2 us | Repeat did not reproduce |
| `projection_128_lt_stack_repeat` | Same route stack; fresh judge run | 1.519x | 20.7 / 18.7 us | No robust paired end-to-end gain |
| `projection_lt_stack_v2` | Above stack + M256 qkv alternate | 1.516x | qkv 18.3 us vs prior 17.6 us | Reject |
| `projection_unpool128_alt2` | Different M128 unpooler algorithm family | 1.499x | 20.6 us; 0.971x of SGLang | Reject |

The only consistent signal is a small speedup for one M128 unpooler plan. It does not materially change the aggregate, and combining it with the pooler descriptor did not reliably improve the score. Keep the public `projection_nathen_v18` submission unchanged.

## Toolchain note

The standard Modal CUDA 12.8 image cannot load this candidate's `libcublasLt.so.13` extension. A CUDA 13 image attempt failed in Modal's image builder before allocating an H200 because the local Modal client runs Python 3.14, unsupported by its 2024.10 image builder. Official judge checks use the compatible runtime and supplied the authoritative timings above. `drv_lt_projection_stack.py` is retained for later use when a matching Modal image is available.
