# Current attention full-context profile

The published attention source, SHA-256 `ffc1c6bb4e95aa30390858eab4c43dbd592b2884b4fe3e54412282eaafdbebaa`, was profiled on Modal H200 with `dev/modal/drv_attention_current_ncu.py`. The driver checks the official reference on B=32, query length 8, context 80,000, page size 1, agentic/coherent data, then profiles a CUDA graph replay. The reference check passed: relative L2 0.03123, bound 0.082.

The captured current `attn_kernel<8, 1, 0>` used 264 blocks of 384 threads, 80 registers per thread and 114.69 KB dynamic shared memory per block. Nsight Compute reported 180.26 µs, DRAM throughput 76.06% of peak, L2 throughput 88.26%, SM throughput 53.56%, and achieved occupancy 36.05%. Registers and shared memory both limited residency to two blocks per SM. This suggests investigating actual transferred bytes and load coalescing before attempting an occupancy-only rewrite; these aggregate counters alone do not establish precisely which accesses are wasteful.

Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file kernels/muse_attention_decode/submission.py --drv dev/modal/drv_attention_current_ncu.py --cmd 'timeout 600s ncu --clock-control none --profile-from-start off --target-processes all --set basic python /w/run/drv.py'`.

Modal app: `ap-gbj78p6vkcIMb9CQbmfTnt`. Clocks were not fixed and Nsight uses replay; this is a diagnostic measurement, not an official score. The older `_os__g_full_verify` profile in `research/ENGINEERING.md` describes a different implementation and should not be attributed to this kernel. Full output is in `ncu.log`.

## Detailed memory follow-up

A second correctness-checked replay (Modal app `ap-W7aAv92aepXABGelrXy6CY`) requested `MemoryWorkloadAnalysis` and `MemoryWorkloadAnalysis_Chart`. The same kernel reported 3.61 TB/s memory throughput, 7.21% L2 hit rate and 2.60% L1/TEX hit rate. This supports the hypothesis that the large cache working set streams mostly from HBM; the hit rates do not by themselves establish the efficiency of coalescing or the amount of avoidable traffic. The chart did not print its detailed metrics, so no sector-efficiency conclusion is available. Nsight's compression speedup suggestion does not establish a practical improvement for the FP8 cache. The full replay output is saved as `memory_ncu.log`.

Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file kernels/muse_attention_decode/submission.py --drv dev/modal/drv_attention_current_ncu.py --cmd 'timeout 600s ncu --clock-control none --profile-from-start off --target-processes all --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart python /w/run/drv.py'`.

## Memory transaction tables

The full `MemoryWorkloadAnalysis_Tables` capture is in `tables_ncu.log` (Modal app `ap-EDS51tdMjviUIl9ejsnLAh`). Reference correctness passed again. Aggregate global loads report 10.94 bytes per 32-byte sector, but that figure includes different instruction types and cannot be ascribed to K/V copies. For the cache-bypass LDGSTS path, the table reports 20,617,048 sectors for 1,288,888 requests, approximately 16.00 sectors/request. A fully active warp copying 16 bytes per lane needs 16 sectors, so this is consistent with efficient K/V copy transactions, although aggregation and inactive lanes prevent a precise per-instruction efficiency calculation. The kernel read 669.45 MB from device memory. This weakens the case for rearranging K/V loads solely to improve coalescing; investigate reduction in necessary traffic or the producer's instruction dependencies instead. Six CTC metrics were unavailable; they do not affect the cited LDGSTS counters. Clocks were unlocked and Nsight replayed the kernel 30 times.

Command: `modal run dev/modal/run_h200.py --problem muse_attention_decode --file kernels/muse_attention_decode/submission.py --drv dev/modal/drv_attention_current_ncu.py --cmd 'timeout 600s ncu --clock-control none --profile-from-start off --target-processes all --section MemoryWorkloadAnalysis_Tables --print-details all python /w/run/drv.py'`.
