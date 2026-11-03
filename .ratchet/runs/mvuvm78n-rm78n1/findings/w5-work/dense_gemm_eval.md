# Dense Projections Component Benchmark on H200 (cuBLASLt Tuning)

## Primary Evidence

- Driver: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_qkv_lt.py`
- Modal Run App: `ap-19J0OIi3WvMGrvsHBQjD2e`
- Hardware: NVIDIA H200 SXM (SM 90, 132 SMs, 141 GB HBM3e)
- Environment: `lmsysorg/sglang:dev-muse-spark-open`
- Raw Log: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_qkv_lt.log`

## Methodology

Evaluated cold-L2 (256 MB buffer flush before each replay) CUDA graph replays for the largest dense matrix product in the prefill layer:
- QKV projection at T=16,384 tokens: Input `[16384, 8192]` BF16, Weight `[2304, 8192]` BF16 -> Output `[16384, 2304]` BF16.
- Total compute: `2 * 16384 * 8192 * 2304 = 6.185e11` FLOPs.
- Evaluated `torch.mm` vs all 8 cuBLASLt heuristic algorithms.

## Measured Timings (µs, median of 10 paired cold-L2 replays)

| Implementation | Configuration / Tile Info | Time (µs) | Speedup vs torch.mm | TFLOP/s | Relative Error vs Ref |
|---|---|---:|---:|---:|---:|
| `torch.mm` | PyTorch standard dispatch | 764.88 | 1.000x | 808.6 | Ref |
| cuBLASLt Algo 0 | `[209, 1, 35, 0, 4, 66, 0, 0, 4, 0]` | 765.78 | 0.999x | 807.7 | 0.000e+00 |
| cuBLASLt Algo 1 | `[24, 1, 35, 0, 4, 66, 0, 0, 7, 0]` | **757.90** | **1.009x** | **816.1** | 0.000e+00 |
| cuBLASLt Algo 2 | `[23, 1, 35, 0, 4, 66, 0, 0, 4, 0]` | 773.81 | 0.988x | 799.3 | 0.000e+00 |
| cuBLASLt Algo 3 | `[31, 1, 35, 0, 4, 66, 0, 0, 7, 0]` | 798.56 | 0.958x | 774.5 | 0.000e+00 |
| cuBLASLt Algo 4 | `[32, 1, 35, 0, 4, 66, 0, 0, 4, 0]` | 824.10 | 0.928x | 750.5 | 0.000e+00 |
| cuBLASLt Algo 5 | `[495, 1, 35, 0, 4, 66, 0, 0, 7, 0]` | 810.62 | 0.944x | 763.0 | 0.000e+00 |
| cuBLASLt Algo 6 | `[197, 1, 35, 0, 4, 66, 0, 0, 2, 0]` | 795.20 | 0.962x | 777.8 | 0.000e+00 |
| cuBLASLt Algo 7 | `[184, 1, 35, 0, 4, 66, 0, 0, 2, 0]` | 787.18 | 0.972x | 785.7 | 0.000e+00 |

## Findings

1. **Compute Saturation**: At 808–816 TFLOP/s, T=16384 QKV operates at >82% of theoretical dense BF16 peak (989.5 TFLOP/s).
2. **Library Tuning Upside is Bounded**: The fastest heuristic algorithm (`Algo 1`, tile id 24) saves only 7.0 µs out of ~9,000 µs whole-layer latency (<0.08% overall speedup).
3. **Challenge Rules Constrain Quantization**:
   - `TASK.md` specifies: *"Weights: ... NVFP4 weights stay NVFP4 and BF16 weights stay BF16, or smaller, losslessly"*.
   - Because BF16 to FP8 is lossy, weights cannot be packed or stored in FP8. Dynamically quantizing weights per call would introduce latency exceeding any theoretical tensor core throughput advantage.
