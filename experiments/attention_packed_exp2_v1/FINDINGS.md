# Packed FP16 exponential for attention probabilities

Hypothesis: each `ex2.approx.f16x2` computes two probability exponentials after rounding the log2 scores to FP16, then packs them to E4M3. QK and PV instructions and their partial accumulators are unchanged from `attention_readable_fp32_path_v1`. Softmax running-max rescaling continues using FP32 exponentials. Both the regular path and its packed-probability overflow check are changed; the original offset logic remains in place.

NVIDIA documents `ex2.approx.f16x2` for SM75 and later and a maximum FP16 relative approximation error of 2^-9.9 (excluding score rounding and final E4M3 quantization): https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2. This is a numerical hypothesis, not a demonstrated error bound on the output. FP16 overflow, masked -infinity, saturation and rounding near E4M3 thresholds need runtime coverage.

Private check `001105f5c7f74313` submitted; result pending. Compare against private check `12167249d81c46ee` on the exact readable parent. Python source parsing passed. This candidate is not promoted; full precision audit, correctness and measured gain are required before any public submission.

CPU sensitivity sweep (`analyze_rounding.py`, `rounding-analysis.txt`): with ideal exp2, FP16 score and result rounding, and nearest-even saturated E4M3 conversion, 553 / 200001 uniformly sampled scores in [-24, 8] changed E4M3 code versus ideal FP32 exp2 before E4M3 conversion. For probabilities >= 2^-9, the largest measured relative error before E4M3 conversion was 0.308%. This is neither the real input distribution nor a simulation of the hardware approximate exp2; it does not establish the output's correctness or a performance gain.

## H200 paired screen: rejected

Judge-image paired cold-L2 CUDA-graph screen, two seeds each for full verification at batches 16 and 32; 16 alternating replays per candidate. Both implementations passed the reference on all four cases. Parent → candidate median µs: 99.0 → 111.0, 180.8 → 203.7, 101.3 → 112.7, 180.0 → 201.6. Paired speedups were 0.887–0.899×, so reject this variant; no inference about other exp approximations or shapes. See `modal.log` and `driver.py`. To reproduce, copy the readable parent's `submission.py` to `dev/modal/_attention_readable_parent.py` before launching `run_judge_image.py`; remove that temporary copy afterwards. Private check `001105f5c7f74313` was still queued at the last poll.

Official private check `001105f5c7f74313` passed at 2.48923× (95% CI 2.45189–2.54378×), below the readable parent check's 2.60414× (95% CI 2.56508–2.64320×). The official result reinforces the local regression; the candidate remains rejected.
