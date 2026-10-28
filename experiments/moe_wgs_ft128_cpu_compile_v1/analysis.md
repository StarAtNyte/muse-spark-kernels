# CPU-only compilation of 128-token WGS candidate

The previously inconclusive `experiments/moe_wgs_ft128_nx1_v1/submission.py` (SHA-256 `2d17be6b48f9043760133479f6a8f8b241e9cda258172c9b36e9040f869f8378`) compiled successfully on CPU-only Modal with `OKC_PRMTSC=1`, matching the flag selected by prior H200 screens. App `ap-FHjEh9yUKkYvotC7cSkuTM` returned code 0 and `COMPILE_DONE`; the trailing client warning concerns final app logs after the result had returned. The first attempt omitted that flag and failed on an unrelated `sc1` compilation path; its setup log is saved separately.

This confirms compilation only: no GPU correctness, timing, or score has been measured. The compiled shared object remained in the ephemeral CPU container, so a GPU benchmark would need to transfer the artifact or recompile; avoid holding an H200 idle for an unbounded build. Probe: `modal run dev/modal/compile_moe_cpu.py --file experiments/moe_wgs_ft128_nx1_v1/submission.py`.
