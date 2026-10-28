# Cooperative single-token routed MoE experiment

Target: M1, hidden/expert width4096, top8, local expert bank0..31. The existing
packed FP4 weight and BF16 GEMV helpers are retained. No dense weight expansion
or activation-dependent preprocessing outside the kernel is introduced.

One occupancy-bounded cooperative grid performs:

1. Read the eight route IDs and derive the local expert list within each CTA.
2. Persistent work over live experts' gate/up tiles, then a grid barrier.
3. Persistent down-projection work, writing BF16 outputs and per-tile weighted
   norm-square partials, then a grid barrier.
4. Two CTAs reduce the norm partials and write disjoint halves of the output.

Every CTA participates in both grid barriers, even with no work. Host dispatch
queries cooperative-launch support and maximum active blocks and never exceeds
the occupancy limit. The grid size depends on hardware, not input values.
Route-dependent work scheduling happens on the GPU inside the timed kernel.

CPU preflight:106 registers,112 bytes static shared memory,4096 bytes dynamic
shared memory, no stack/spills. This is not yet a correctness/performance claim.
`screen.log` records independent task checks, mutation of routing during graph
replay, and paired cold-L2 full-pipeline timings against the current M1 kernel.
The synchronization and persistent scheduling overhead may exceed the saved
launch/routing overhead; the experiment measures that tradeoff.

CUDA cooperative-grid API/occupancy requirements:
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html

Outcome: runtime and CUDA graph capture work on H200. All24 timed task checks
passed, with outputs identical to the baseline. Two valid route mutations
(single local slot7 and all8 local experts) passed the task checker after
replaying an existing graph. A separate no-local input returned exact zero;
the judge checker excludes that input domain and was not counted as a pass.

Performance: the512-block configuration without prefetch roughly matches the
baseline; smaller caps and down-weight L2 prefetch lose. No promotion or
official submission. Cooperative barriers/scheduling do not produce a useful
latency reduction in this design. `zero_domain_check.log` is the earlier test
wrapper failure on the checker's excluded input, not a kernel failure.
