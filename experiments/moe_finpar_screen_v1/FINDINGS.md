# WGS parallel final pass (sums of squares in phase-B epilogue) — correct, not adopted

Candidate `candidate.py` (`OKC_WGSFINPAR`: warp-reduced atomics of each route's sum of squares in the phase-B
epilogue, column-parallel single-pass final over (token, 1024-column segment)). Note: the existing
`OKC_WGSFINSS=0` path is stale in WGS builds (does not compile). Paired screen (`dev/modal/drv_moe_pair.py`,
6 seeds, all outputs reference-checked incl. bs 2-5): bs 1-16 unchanged (WGS not used), bs32 1.012x, bs128 0.989x,
bs256 0.980x. Epilogue atomics cost more than the faster final pass saves at larger batches; a bs<=32-only switch
would be worth ~+0.2% on the MoE score (below judge noise). Candidate module renamed (v216fp) so base/candidate
load_inline names do not collide in one process.
