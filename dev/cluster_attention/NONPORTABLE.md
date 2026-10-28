# Nonportable 16-block Hopper cluster

The eight-block DSM kernel gave good batch-16 results, but each block still
loops over about five 64-key tiles for a 2048-token window plus seven new-query
positions. A sixteen-block cluster cuts that to at most three tiles. Its merge
assigns four output rows to each rank; only the first 64 threads participate in
that merge, with every thread reaching both cluster barriers.

The CUDA API requires an explicit nonportable-cluster opt-in. CPU compilation
passed; the H200 runtime occupancy query reported 28 active clusters. This is
measured support on this H200, not a promise for other SKUs or MIG layouts.

`nonportable_screen.log`: all 16 candidate task checks and 2 edge/scaling checks
passed. Batch 1 gained 1.105× / 1.092× on two seeds; batch 4 gained 1.039× /
1.050×. Batch 16 lost (0.769× / 0.777×). Batch-32 production calls an unavailable
library in this image and its ~8 µs timing is invalid; ignore that comparison.
It does not affect the validity of the candidate's independent correctness
checks or the small-batch custom paths. The confirmation screen explicitly
requires a tensor baseline and independently checks its correctness.

`nonportable_prepare_compile.log`: fused-prepare sixteen-block kernel compiles
at 118 registers without spills (32 bytes stack). Only ranks 0..7 prepare one
query token each and publish new KV, avoiding duplicate writes; the remaining
ranks read prepared queries from those eight ranks after a cluster barrier.
A second barrier protects the shared query fragments before attention reuses
that storage. Distinct clusters own distinct batch/head cache regions.

`nonportable_confirm.log`: fresh seeds 5011/15241/49157, batches 1/2/4/8,
production validation, separate preparation, fused preparation and raw
softmax-statistics merge. Edge/scaled-cache checks are mandatory. All page
lookups, queries, KV updates, attention, cluster joins and output normalization
are inside the timed graph. No keys or values are omitted.

Primary reference:
https://docs.nvidia.com/cuda/archive/12.8.1/pdf/Hopper_Tuning_Guide.pdf

## Confirmation outcome

All 54 checks passed: 12 production outputs, 36 candidate outputs, and 6 edge /
scaled-cache outputs. Separate-prepare batch-1 ratios were 1.068/1.111/1.095×;
batch-4 ratios were 1.041/1.032/1.045×. Batch 8 was effectively tied. Fused
preparation and raw statistics did not consistently lower absolute latency,
so retain the simpler separate preparation and gate sixteen-block clusters to
batches 1..4. All larger batches retain production dispatch.

`build_nonportable_submission.py` builds a complete standalone source from the
immutable `attention_fp8p_cluster_v1` parent. It is undergoing full official
validation as `attention_cluster16_small_v1`; no result is claimed yet.

## Official result: do not promote yet

Private `389d845575f84f5f` passed all 16 tests and 12 timings. The targeted
batch-1 sliding verify case fell from 13.656 to 12.729 µs across the two
private runs (6.8%). However, aggregate speedup was 1.726756× (CI
1.700855–1.752657), below the existing 1.732609× best. Unchanged paths also
varied across runs. The small structural local gain is credible, but this run
does not establish a better full score. Retain the candidate for combination
with a larger optimization; no public submission or promotion.
