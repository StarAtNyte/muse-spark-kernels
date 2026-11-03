# WILD

Before each WILD experiment, write 10 unfiltered ideas here, then pick the one least like anything in `results.tsv`.

## 2026-11-03 batch
1. SMOE: dynamic global unit claiming for M<=8 MoE (fast SMs take more units), expert-ordered claims so down-proj waits vanish. (PICKED, in progress: experiments/moe_smoe_dyn_v1)
2. Decoder: prefetch the MoE's first ring of expert weights during the router kernel via PDL (routing unknown -> prefetch most-popular local experts speculatively into L2).
3. Merge router + MoE into one persistent kernel at T=1 (router GEMV result in-CTA, no launch boundary).
4. Attention decode: one persistent kernel that also does the o-projection on the last-finishing CTAs (fusion across attention/o).
5. MoE bs16/32: claim-based scheduling like SMOE with 64-row WGS units instead of static stream-K ranges.
6. Split phase B per expert into cluster-of-2 DSMEM exchange of h so B can start before global h is written.
7. Use L2 persistence window (cudaAccessPolicyWindow) for hbuf/ybuf scratch so they never hit HBM.
8. Decoder T=1: run pooler/unpooler projections concurrently on a second stream branch inside the graph with MoE.
9. Projection: weight-stationary across graph replays is illegal; instead do K-split with 2 CTAs/SM to raise memory-level parallelism.
10. Verify-8 attention: process all 8 query tokens with one wgmma M=64 tile across heads (GQA packing) to cut KV reads to one pass.
