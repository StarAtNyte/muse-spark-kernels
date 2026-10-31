# Forge prefill H200 smoke pass

Forge job `1793428292` ran `notes/smoke_m0gate.py` and returned `PROBE PASS max_diff=n/a`. It checks successful launches and output shapes/dtypes for 128, 753 and 3012 rows; it does not compare output values or measure performance. The source is workspace commit `0df0d06`, SHA-256 `839f34e9f6e13bb0bfa04ee6be1cc338f93802bf91fb9ae94c479915b56bccad`.

Relative to the previous smoke probe, this commit adds only `// fk-cuda-cflags: -gencode=arch=compute_90a,code=sm_90a` to the kernel source. The previous job failed at its first PyTorch input operation with a CUDA no-kernel-image error (see `experiments/forge_prefill_iter2_probe_v1/`). The retry passing supports a target mismatch, although the earlier controlled minimal extension test did not reproduce that failure, and the two jobs used separate containers.

The candidate has not yet passed numerical correctness or timing, and is not promoted. The official full-layer judge remains necessary.
