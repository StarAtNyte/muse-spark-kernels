# Prefill Layer Optimization Log (`muse_prefill_layer`)

- Target Board: `muse_prefill_layer`
- Current Public Score: **1.7874x** (Job: `68ac9af85a3a4e3d`, Rank 123)
- Leaderboard #1: **2.0325x** (`zzz0906`), Top-50 threshold: ~1.81x
- Evaluation Hardware: NVIDIA H200 SXM, CUDA 12.8 / 13.0, cold L2 before replay

## 1. Fused Prefill K64 MoE Path Screen

- **Hypothesis**: Reordering NVFP4 codes and activations with a 64-element K stage permutation enables hoisting block scale decoding across 4 inner WGMMA steps.
- **Experimental Screen** (`experiments/prefill_k64_fused_v1/`, Modal H200 app `ap-0O3q4b5F90gJRHLW9Rp4i4`):
  - Correctness: relative L2 difference to parent ~1.4–1.7e-4 across 32, 96, and 753 tokens.
  - Paired cold-L2 full-layer timings (Parent vs Candidate):
    - `batch 1, q 753, sliding, ctx 80000`: 513.8 µs vs 512.4 µs (1.0028x)
    - `batch 4, q 753, full, ctx 80000`: 2980.0 µs vs 2982.3 µs (0.9992x)
    - `batch 1, q 16384, full, ctx 0`: 9144.9 µs vs 9191.0 µs (0.9950x)
    - `batch 1, q 16384, full, ctx 65536`: 15829.2 µs vs 15663.2 µs (1.0106x)
  - Geometric mean speedup: **1.0019x** (+0.19%, within ±1.5% measurement noise).
- **Decision**: **Archive only**. The cross-thread shared memory exchange and barrier during activation gather counteract the register savings from scale reuse. In accordance with the prompt guidance (*"officially check the fused prefill K64 MoE path if its screens show a win"*), an official check was avoided to preserve check budget.

## 2. Attention Component Evaluation (Custom Kernel vs FlashAttention-3)

- **Driver**: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_attn_bench.py`
- **Experimental Screen** (Modal H200 app `ap-r23oyq3FYAY4C9Br6zc5rz`):
  - Evaluated cold-L2 graph replays across all 8 scored prefill cases.
  - Results (Custom Hopper kernel `attn_fwd_kernel` vs best FA3 configuration):
    - `turn1_slide`: 31.1 µs vs 36.7 µs (**1.18x** faster)
    - `turn1_full`: 367.6 µs vs 429.4 µs (**1.17x** faster)
    - `turn4_slide`: 68.7 µs vs 72.3 µs (**1.05x** faster)
    - `turn4_full`: 1403.6 µs vs 1640.9 µs (**1.17x** faster)
    - `ft_full_0`: 837.0 µs vs 930.0 µs (**1.11x** faster)
    - `ft_slide_32k`: 309.6 µs vs 334.4 µs (**1.08x** faster)
    - `ft_full_32k`: 3831.3 µs vs 4446.7 µs (**1.16x** faster)
    - `ft_full_64k`: 6772.8 µs vs 7974.1 µs (**1.18x** faster)
- **Conclusion**: The incumbent custom Hopper kernel (`attn_fwd_kernel`) is superior to FA3 across all shapes due to in-register fused QK-norm, RoPE, and `rms(att) * sigmoid(gate)` epilogue. Replacing custom attention with FA3 would cause regressions.

## 3. Dense Projection Component Evaluation (cuBLASLt Tuning)

- **Driver**: `.ratchet/runs/mvuvm78n-rm78n1/findings/w5-work/test_qkv_lt.py`
- **Experimental Screen** (Modal H200 app `ap-19J0OIi3WvMGrvsHBQjD2e`):
  - Evaluated T=16384 QKV dense projection (`torch.mm` vs 8 cuBLASLt heuristic algos).
  - `torch.mm`: 764.88 µs (808.6 TFLOP/s, ~82% peak SM90 BF16 tensor core rate).
  - cuBLASLt Algo 1: 757.90 µs (816.1 TFLOP/s, 1.009x speedup, delta 7.0 µs).
  - All other cuBLASLt algos regressed (773–824 µs).
- **Constraint**: Under challenge rules, dense weights and activations cannot be quantized to FP8 losslessly.

## 4. Overall Decision and Next Steps

- Following the constraint (*"submit only when the CI beats 1.7874x"* and *"This board is separate and does not count toward overall rank, so keep the effort bounded"*), no submission was made since measured improvements in component screens were bounded below the +1.5% CI threshold.
- Future investigations should explore fused dense projection pipelines or warp-specialized GEMM + boundary fusions where compute and memory roundtrips can be eliminated end-to-end.
