# Attention Component Benchmark on H200 (Own Kernel vs FA3)

## Primary Evidence

- Driver: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_attn_bench.py`
- Modal Run App: `ap-r23oyq3FYAY4C9Br6zc5rz`
- Hardware: NVIDIA H200 SXM (SM 90, 132 SMs, 141 GB HBM3e)
- Environment: `lmsysorg/sglang:dev-muse-spark-open`
- Raw Log: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_attn_bench.log`

## Methodology

Evaluated cold-L2 (256 MB buffer flush before each replay) CUDA graph replays across all 8 scored prefill shapes comparing:
1. `own`: Custom Hopper attention kernel (`attn_fwd_kernel`) with fused norm-gate epilogue (variant 8 unpipelined for sliding, variant 43 pipelined for full).
2. `fa3_default`: SGLang FlashAttention-3 (`sgl_kernel.flash_attn.flash_attn_with_kvcache`) with `num_splits=0` + separate fused norm-gate kernel.
3. `fa3_ns1`, `fa3_ns2`, `fa3_ns4`: FlashAttention-3 with explicit split-KV counts (1, 2, 4) + separate fused norm-gate kernel.

## Measured Timings (µs, median of 8 paired cold-L2 replays)

| Scored Shape | Weight | Own Kernel (µs) | FA3 Default (µs) | FA3 Best Splits (µs) | Own vs FA3 Best |
|---|---|---:|---:|---:|---:|
| `turn1_slide` (B=1, L=753, ctx=80k, slide) | 46 | **31.1** (v8 ns2) | 37.8 | 36.7 (ns4) | **1.18x** |
| `turn1_full` (B=1, L=753, ctx=80k, full) | 16 | **367.6** (v43 ns2) | 429.8 | 429.4 (ns2) | **1.17x** |
| `turn4_slide` (B=4, L=753, ctx=80k, slide) | 46 | **68.7** (v8 ns1) | 72.7 | 72.3 (ns1) | **1.05x** |
| `turn4_full` (B=4, L=753, ctx=80k, full) | 16 | **1403.6** (v43 ns1) | 1642.3 | 1640.9 (ns1) | **1.17x** |
| `ft_full_0` (B=1, L=16384, ctx=0, full) | 16 | **837.0** (v43 ns1) | 930.9 | 930.0 (ns1) | **1.11x** |
| `ft_slide_32k` (B=1, L=16384, ctx=32k, slide) | 138 | **309.6** (v8 ns1) | 335.6 | 334.4 (ns1) | **1.08x** |
| `ft_full_32k` (B=1, L=16384, ctx=32k, full) | 16 | **3831.3** (v43 ns1) | 4522.2 | 4446.7 (ns4) | **1.16x** |
| `ft_full_64k` (B=1, L=16384, ctx=64k, full) | 16 | **6772.8** (v43 ns1) | 8105.5 | 7974.1 (ns2) | **1.18x** |

## Findings

1. **Own kernel dominates FA3 across all shapes**: Across all 8 cases, the incumbent custom Hopper kernel (`attn_fwd_kernel`) is consistently 5% to 18% faster than FlashAttention-3.
2. **Epilogue fusion advantage**: FA3 requires materializing intermediate BF16 output tensors into DRAM and then dispatching `_norm_gate_kernel`, incurring substantial DRAM read/write roundtrip bandwidth. In contrast, `attn_fwd_kernel` performs QK-norm, RoPE, FP8 attention, and the post-norm gate epilogue entirely in registers and shared memory.
3. **Splitting behavior**: For FA3, tuning `num_splits` offers only negligible improvements (at most 1–2%) and never matches the fused kernel.
