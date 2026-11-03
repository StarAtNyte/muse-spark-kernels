# Decoder FP32-output fallbacks, including small-token pooler

V2 changed the decoder's dense and direct QKV PyTorch GEMM fallbacks, but omitted the `T <= 8` pooler `F.linear` path in `_pool_router_fp32`. V3 changes that call to an FP32-output matmul followed by the final BF16 cast. An AST scan of remaining Python `@` operators found only FP64 reference/preparation calculations; CUDA and dynamically generated/library code still require separate review.

Private check `f08d0cc2e617467e` is pending. Syntax passed. This does not establish complete compliance or performance; retain V2's pending result for provenance, not as validation of this source.

Official private check `f08d0cc2e617467e` passed 13/13 at 2.23061× (95% CI 2.19715–2.26407×). A later audit found the main WGS MoE may store completed per-expert values in BF16 before final combination; whether that is allowed is unresolved. V4 tests FP32 storage before submission.
