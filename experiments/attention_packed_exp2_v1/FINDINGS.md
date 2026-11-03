# Packed FP16 exponential for attention probabilities

Hypothesis: each `ex2.approx.f16x2` computes two probability exponentials after rounding the log2 scores to FP16, then packs them to E4M3. QK and PV instructions and their partial accumulators are unchanged from `attention_readable_fp32_path_v1`. Softmax running-max rescaling continues using FP32 exponentials. Both the regular path and its packed-probability overflow check are changed; the original offset logic remains in place.

NVIDIA documents `ex2.approx.f16x2` for SM75 and later and a maximum FP16 relative approximation error of 2^-9.9 (excluding score rounding and final E4M3 quantization): https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2. This is a numerical hypothesis, not a demonstrated error bound on the output. FP16 overflow, masked -infinity, saturation and rounding near E4M3 thresholds need runtime coverage.

Private check `001105f5c7f74313` submitted; result pending. Compare against private check `12167249d81c46ee` on the exact readable parent. Python source parsing passed. This candidate is not promoted; full precision audit, correctness and measured gain are required before any public submission.
