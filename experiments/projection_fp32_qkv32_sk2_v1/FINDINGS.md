# QKV m=32 alternate projection backend

The FP32-reduction projection parent scored qkv m=32 at 14.42 µs, matching the paired SGLang reference on that run. This candidate routes that case to the existing vjeux backend, which includes the FP32-split SK2 kernel used for qkv m=16. Its `prepare` may select other internal implementations, so the eventual profile must confirm which path actually ran. All other dispatch entries and the disabled BF16 split paths are unchanged.

Private check `e4df402a11e94f52` is pending. The parent passed 9/9 at 1.53642×. Python syntax passed; neither a gain nor complete compliance is claimed. Compare paired per-shape timing and the profile, and audit the selected path before any public submission.

Official private check `e4df402a11e94f52` passed 9/9 at 1.54234× (95% CI 1.51920–1.56547×). Its profile confirms `sk2_kernel` selected for qkv m=32, with paired-reference latency 10.74 µs versus 14.42 µs in the parent check. The total parent score was 1.53642× (CI 1.51338–1.55947×): separate runs and overlapping intervals do not establish an overall score gain. The code path's in-kernel reductions use FP32, but the full submission remains under precision audit. No public submission has been made; a same-run paired screen is still needed.
