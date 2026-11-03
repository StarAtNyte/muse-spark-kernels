# FP32-output projection fallbacks

Builds on the corrected compact QKV candidate. Four reachable GPU fallback calls that previously returned BF16 directly (`_v12`, vjeux `F.linear`, `torchT`, and the v6 transposed-matmul route) now request FP32 matrix-multiplication outputs and cast once to BF16. This avoids relying on PyTorch's default BF16 reduced-precision-reduction policy. It does not change global PyTorch settings or the optimized custom routes.

Private check `6dea36df13e74a6a` is pending. Python parsing passed. This is a targeted mitigation; other custom and library paths still require a complete precision audit. Compare performance against v3 and verify actual fallback selection before promotion.
