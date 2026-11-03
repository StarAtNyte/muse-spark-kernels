# Compact FP32-split QKV transfer to projection

Transfers the readable compact QKV kernel and packing code from `decoder_readable_fp32_reduction_v1` into `projection_fp32_qkv32_sk2_v1`. For QKV m=128 on H200 it uses a cluster reduction that exchanges and sums `float4` partials before the final BF16 store; all other cases fall back to the parent. It compiles only the existing selected configuration. If preparation cannot build it, the parent remains available.

Private check `7ecb1b41c2444395` submitted; correctness and speed pending. Source parses. The full submission still needs a precision and source audit before public use. In particular, check fresh-thread CUDA context binding on the new fast path; the original parent wrapper binds context before dispatch.
