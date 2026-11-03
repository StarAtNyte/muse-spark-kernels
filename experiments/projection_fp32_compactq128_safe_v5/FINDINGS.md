# Align projection tuner with FP32-output fallback

V4 changed runtime BF16 GEMM fallbacks to request FP32 results, but the chenwainuo cuBLASLt tuner still timed a BF16-output `torch.mm` as its comparison. V5 times the same FP32-output matmul plus final BF16 conversion used at runtime, so it can select the faster compliant library path under actual costs. This affects in-prepare selection, not the mathematical computation. The selected cuBLASLt compute mode is FP32; dynamically generated split-K plans enforce compute-type reduction.

Private check `e62a07b64045453a` is pending. Python syntax passed; no correctness or performance result is claimed. V4 check `6dea36df13e74a6a` and compact-only v3 check are also pending. Full source and precision review remains necessary.

Official check `e62a07b64045453a` passed 9/9 at 1.54735× (95% CI 1.52414–1.57056×). The profile confirms SK2 at QKV-32 (10.57 µs) and the compact kernel at QKV-128 (14.18 µs). The prepare log reports 973 off-score routes from 1,738 timed cells; the simpler `../projection_fp32_compactq128_nov6_v6` is being tested before public submission. Source and precision review remains open.
