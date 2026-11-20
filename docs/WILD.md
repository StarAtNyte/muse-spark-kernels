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

## 2026-11-04 batch
1. Attention q_len 1 on the FP8 wgmma PV path used by q_len 8 (P in E4M3, 16 real rows padded to M=64): one P format for both q_lens, and it settles audit question U1 (FP16 P). (PICKED; branch wild/attn_l1_fp8pv, design only, not implemented)
2. Decode layer: one persistent cooperative kernel with a device-side phase graph (attn -> o-proj -> router -> MoE) replacing PDL launch chains.
3. Store the FP8 KV cache K-transposed per 64-key tile at write time so QK needs no ldmatrix transpose (new-token write path pays once).
4. MoE: sort tokens by expert on-device and run one wgmma GEMM per expert with M padded to 8 (expert-major instead of token-major).
5. Router: compute top-k with a bitonic network inside the projection epilogue (no separate router launch).
6. Attention: assign KV pages to CTAs by page-table locality (physical row proximity) rather than logical order.
7. Scratch buffers (hbuf/ybuf) in distributed shared memory across a cluster instead of global memory, for T<=8.
8. Two-graph pipelining: overlap layer N's MoE tail with layer N+1's QKV head via separate streams (only legal if the harness graph keeps both nodes).
9. Dequant NVFP4 once per CTA into FP8 smem and reuse across the 8 verify tokens (weights read once, math in FP8 wgmma).
10. Sliding-window attention: keep the last 2048 keys' FP16 copy in a side cache updated at write time (violates "no caching derived from data"; rejected on sight).

## 2026-11-04 evening: 10 unfiltered ideas
1. Attention B=1: one-CTA-per-SM persistent kernel that the projection kernel before it pre-launches via PDL, spinning on a ready flag so the attention launch latency vanishes.
2. Attention B=1 full: split-KV merge with a 16-CTA non-portable cluster over DSMEM, with no global atomics and no last-CTA pass.
3. MoE: co-schedule w13 and w2 for different experts in the same wave (expert i's w2 streams while expert i+1's w13 streams), with no phase A/B split.
4. MoE: repack NVFP4 in prepare so each SM's weight slice is one contiguous multi-MB region, then a single bulk TMA stream per SM with zero unit boundaries.
5. MoE dequant on the FMA pipe: HFMA2 magic-number e2m1 conversion with the 2^126 exponent bias folded into the block scale, balancing the ALU/FMA pipes.
6. Decoder: run the MoE w13 stream for the shared hot experts before routing finishes, speculating that the top experts from the previous token's routing are reused, with exact fallback (risky: maybe a rule issue, since it is input-dependent work).
7. Decoder: fuse the unpooler with the next layer's nothing; instead start the o-proj weight stream into L2 during attention (L2 prefetch rejected before; try the TMA prefetch issued by idle SMs only).
8. Projections: 11-bit codec (narrower exponent window, more escapes) for pooler/unpooler at m>=16, where bytes dominate.
9. Attention sliding: compute QK for 2048 tokens with 1 KV head per CTA cluster of 2, exchanging partial softmax stats via DSMEM, aiming at a ~2 us total.
10. Whole decoder T=1 as a single persistent megakernel with a task-queue of (kernel, tile) items and per-tile dependency counters (keell0renz measured cross-SM flags as slow under saturated HBM; T=1 is latency- not BW-bound, so maybe different).
Pick: #2 (cluster DSMEM merge for attention B=1). Least like results.tsv (no cluster-merge run logged). Overlaps the running attn_b1_v2 agent's brief, so its result counts as the wild run; if the agent skips it, run it as wild/attn_b1_cluster_merge once an agent slot frees.
