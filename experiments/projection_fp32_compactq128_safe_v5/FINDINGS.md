# Align projection tuner with FP32-output fallback

V4 changed runtime BF16 GEMM fallbacks to request FP32 results, but the chenwainuo cuBLASLt tuner still timed a BF16-output `torch.mm` as its comparison. V5 times the same FP32-output matmul plus final BF16 conversion used at runtime, so it can select the faster compliant library path under actual costs. This affects in-prepare selection, not the mathematical computation. The selected cuBLASLt compute mode is FP32; dynamically generated split-K plans enforce compute-type reduction.

Private check `e62a07b64045453a` is pending. Python syntax passed; no correctness or performance result is claimed. V4 check `6dea36df13e74a6a` and compact-only v3 check are also pending. Full source and precision review remains necessary.
