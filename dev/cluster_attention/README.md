# Cluster attention experiment

This is an experimental architecture, not a promoted submission.

The target is eight-token sliding attention. Eight CTAs per KV head compute
disjoint contiguous key ranges. Each CTA writes its 64x64 normalized partial
and 64 log-sums into its own shared memory, reusing attention staging storage.
After a cluster barrier, the CTAs divide the 64 output rows between them and
read the eight partials with Hopper distributed shared memory. They merge,
round attention to BF16, normalize, gate, and write the final output directly.
The ordinary Q/K preparation and cache-update launch precedes attention, so
there is no cross-CTA race on newly appended cache entries.

`build_fused.py` derives the attention loop from the FP8 joint-layout candidate.
It retains every key, the FP32 softmax denominator, and the production BF16
rounding before RMS normalization. It removes global partial writes and the
standalone merge launch. The portable eight-CTA cluster requirement limits
scheduling flexibility; the launch savings must exceed that cost.

CPU compilation: 119 registers, no stack, no spills. GPU end-to-end performance
and task correctness are recorded in `fused_run.log`; compilation alone is not
evidence of a speedup.

The preceding `probe.py` tested DSM merging independently. All six cases passed,
including empty splits/rows and epsilon-dominated tiny values. Cold-L2 graph
latencies, including replay overhead: DSM 9.54–12.06 us versus a Triton global
merge 6.45–10.70 us. This rejects DSM as a standalone merge replacement; only
integration can justify it. That probe folds the denominator into RMS epsilon
algebraically and does not model intermediate BF16 rounding. The integrated
kernel deliberately preserves that rounding instead.

Integrated small-batch screen (`fused_run.log`): all eight independent task
checks passed. Two seeds per setting, page64 edges/scaled-soft cases and
page1 long-context coherent cases. Batch1 paired median speedups
1.007/1.013/1.007/1.029; batch4 1.074/1.052/1.050/1.050. These modest local
gains do not justify a leaderboard claim. `drv_cluster_large.py` screens the
larger batches against real FA3 plus a clearly identified Triton norm/gate
proxy; only official evaluation can establish the final aggregate improvement.

Size sweep (`sizes_run.log`): 24/24 task checks passed, two fresh seeds per
batch and cluster size. Cluster8 wins at batch1 (~19.4 us) and16 (~25.1 us).
Cluster4 wins at batch32 (~33.7 us), versus cluster8 ~38.5 us. Cluster1/2
are substantially slower; reject those geometries.

Official private experiment `attention_cluster_sliding_v1`, job
`4448135fa8d44951`, isolates cluster8 to sliding verification batch2..24.
Batch1 and batch32 retain production dispatch because the available evidence
does not establish an improvement there. No production promotion yet.

Official outcome: full correctness coverage passed, aggregate1.7062347194x
versus production private1.7096881132x. Targeted verify16/sliding latency
19.70 us versus22.40 us in production's earlier evaluation (about12% lower).
The aggregate does not establish an improvement; no promotion. This is a
useful specialized branch to combine with separately validated full-context
improvements. The local FA3+norm proxy overstated the size of the advantage,
as anticipated; use official measurements for decisions.

Next experiment: `build_prepare_fused.py` removes the preceding preparation
launch. Muse's eight verify tokens match the eight cluster ranks. Each rank
prepares one token's eight query heads and its KV head's new K/V. After a
cluster barrier, ranks gather the eight 512-byte query fragments through DSM.
A second cluster barrier protects those source fragments from early reuse.
The original all-key attention loop and BF16-rounding epilogue then execute.

The synchronization argument depends on disjoint (batch,KV-head) clusters:
each cache region is written and read within one cluster, and the new token
slots are disjoint. CUDA cooperative-group sync publishes prior memory
accesses to participating threads; no ordering across clusters is assumed.
See https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#sync

The performance hypothesis is narrower: remove one launch and intermediate
query writes/reads without duplicating all eight tokens' preparation in every
rank. The costs are two additional cluster barriers, DSM query broadcasts,
and less preparation parallelism. Correctness and timing must establish
whether the exchange is worthwhile; source lives in `prepare_fused.py` and
the independent task-check driver in `../modal/drv_cluster_prepare.py`.
