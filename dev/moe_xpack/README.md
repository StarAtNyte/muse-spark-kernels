# One-time routed activation packing

Larger dataflow experiment, frozen v22. Keep exact NVFP4 weights/scales, BF16 activations and FP32 accumulation. After routing, gather each active 32-token expert chunk once into `[route item, K64 chunk, token fragment, lane, 8 BF16]`, matching the row-split shared-memory activation layout. All output CTAs then issue contiguous activation copies instead of repeatedly resolving route pointers and gathering individual token rows. The second stage can independently pack gate/up outputs in the same format.

All packing is inside the timed kernel_fn graph. The added allocations, launches, full padded packed writes and GEMM reads are timed. This is neither cached input data nor dequantized weight storage. Original path and selected packed stages share the same compiled module and unmodified routing/finish. Packed storage is fresh on every call, overwritten on every replay; no cross-call reuse.

Tradeoff: substantial padded activation write traffic and additional kernel launches. Test gate/up only, down only, and both packed; six inputs across M128/M256, three seeds each, 18 independent reference/bit-equality checks and 24 alternating cold-L2 pairs each. Diagnostic bundle is not a submission.

Evidence: compile.log, screen.log, ../modal/drv_moe_xpack.py. Build script build.py.

## Outcome

All 18 independent reference checks passed, outputs bit-identical. Geomean ratios at M128: both0.9701, gate/up0.9757, down0.9851. M256: both0.9434, gate/up0.9706, down0.9744. Reject separate packing. No official quota spent; GPU job completed. Follow-up dev/moe_xfused eliminates the down-input pack kernel by writing the new layout directly in gate/up's epilogue.
