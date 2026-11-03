# Compact QKV with thread-context binding

Identical to `projection_fp32_compactq128_v1` except it calls the inherited `_ctx_bind(data["x"])` before the compact launch, preserving the parent's fresh-thread CUDA context behavior. The v1 private check tests a different source; this v2 has only been syntax-checked and needs its own correctness and performance validation. It is held pending the v1 performance result and full precision review.

V2 also lacks the module-level `torch` import and fails on import. Use v3 for further testing.
