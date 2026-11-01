# MoE dry-segment instruction-cache warm-up (2026-11-01): built, correct, 15-18% slower. Closed.

Candidate: `wgs_seg` takes a runtime `dry` flag (skips the two stage waits, the slot/x-slot arrivals and the epilogue; one
stage; result kept alive by the existing `d[0][0]==1234.5f` trick). `wgs_phase<false>` runs its first segment once dry, then
again for real (`-DOKC_DRYW=1`; default 0 builds the same code as before). Paired, read-flush, file's own `kernel_fn`, 40 pairs:

| bs | seed | main us | dry us | ratio |
|---|---|---|---|---|
| 32 | 3357109366520139634 | 118.0 | 140.0 | 0.843 |
| 32 | 419 | 119.3 | 140.3 | 0.850 |
| 32 | 7919 | 134.0 | 160.2 | 0.836 |
| 16 | 3357068220476696277 | 97.0 | 96.8 | 1.003 |
| 128 | 11 | 168.7 | 205.0 | 0.823 |
| 64 | 5 | 155.2 | 185.9 | 0.835 |

All outputs pass the correctness check. The +22 us at every WGS size is far more than one stage of work, so making `dry` a
runtime value inside the unrolled stage body (branches around the waits/arrivals) most likely costs the wgmma pipeline
(ptxas serialization, C7512) rather than the dry stage itself. A compile-time `dry` would duplicate the 8 n-tile code copies.
Conclusion: the ~1% prize for the 32-token cold penalty is not reachable with this structure. Do not retry without a
compile-time-specialised, single-n-tile-variant design.
