# QKV m=32 alternate projection backend

The FP32-reduction projection parent scored qkv m=32 at 14.42 µs, matching the paired SGLang reference on that run. This candidate routes that case to the existing vjeux backend, which includes the FP32-split SK2 kernel used for qkv m=16. Its `prepare` may select other internal implementations, so the eventual profile must confirm which path actually ran. All other dispatch entries and the disabled BF16 split paths are unchanged.

Private check `e4df402a11e94f52` is pending. The parent passed 9/9 at 1.53642×. Python syntax passed; neither a gain nor complete compliance is claimed. Compare paired per-shape timing and the profile, and audit the selected path before any public submission.
