# Decoder FP32-output projection fallbacks

Based on the 13/13-passing readable decoder, change its two BF16-output PyTorch projection paths (the bfp backend's `F.linear` fallback and the direct QKV `torch.mm`) to request FP32 matmul outputs and cast once to BF16. This avoids relying on PyTorch's default allowance for reduced-precision BF16 GEMM reductions, without modifying global settings or the other dispatch routes. It addresses a possible rule risk, not a demonstrated reduced-precision algorithm in the prior profile.

Private full decoder check `5efc65e1619a483c` is pending. The source parses; performance, full correctness, and remaining custom/library precision paths require validation before public submission.
