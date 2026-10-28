# Current attention full-context profile

The published attention source, SHA-256 `ffc1c6bb4e95aa30390858eab4c43dbd592b2884b4fe3e54412282eaafdbebaa`, was profiled on Modal H200 with `dev/modal/drv_attention_current_ncu.py`. The driver checks the official reference on B=32, query length 8, context 80,000, page size 1, agentic/coherent data, then profiles a CUDA graph replay. The reference check passed: relative L2 0.03123, bound 0.082.

The captured current `attn_kernel<8, 1, 0>` used 264 blocks of 384 threads, 80 registers per thread and 114.69 KB dynamic shared memory per block. Nsight Compute reported 180.26 µs, DRAM throughput 76.06% of peak, L2 throughput 88.26%, SM throughput 53.56%, and achieved occupancy 36.05%. Registers and shared memory both limited residency to two blocks per SM. This suggests investigating actual transferred bytes and load coalescing before attempting an occupancy-only rewrite; these aggregate counters alone do not establish precisely which accesses are wasteful.

Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file kernels/muse_attention_decode/submission.py --drv dev/modal/drv_attention_current_ncu.py --cmd 'timeout 600s ncu --clock-control none --profile-from-start off --target-processes all --set basic python /w/run/drv.py'`.

Modal app: `ap-gbj78p6vkcIMb9CQbmfTnt`. Clocks were not fixed and Nsight uses replay; this is a diagnostic measurement, not an official score. The older `_os__g_full_verify` profile in `research/ENGINEERING.md` describes a different implementation and should not be attributed to this kernel. Full output is in `ncu.log`.
