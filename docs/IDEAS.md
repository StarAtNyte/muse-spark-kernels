# IDEAS

Ranked backlog. Columns: expected gain, cost, status. History lives in `research/experiment-priorities.md` and `research/NEXT-MOE-REWRITE.md`.

## Pending scored submission

- decoder `experiments/decoder_compact_tinywarp_stack_v1`: checked 2.26054x (CI 2.227-2.294), scored quota was exhausted at check time. Tied with best (2.2636x); submit only if decoder scored quota is spare.

## Ranked

| idea | problem | expected gain | cost | status |
|---|---|---|---|---|

## Wild

(seed list; add design-changing ideas here)

## 2026-11-03 public-repo refresh
Top-10 teams (dolanzhao, tonybolivar, pjmullin3, tjiang-fair, gogowhy, olety) have no public OKC repos. New public repos (Mithun-033 1.255x proj, AutoclickerI 1.89x decoder) are below our scores. Nothing to port. Ideas from our own floors work instead:

| idea | problem | expected gain | cost | status |
|---|---|---|---|---|
| SMOE dynamic-claim M<=8 MoE | moe, decoder | MoE +5-8%, decoder +2-3% | 1 day | screening (moe_smoe_dyn_v1) |
| Claim-based scheduling for WGS bs16/32 | moe, decoder | +3-6% MoE | 1 day | after SMOE |
| PDL-overlapped SMOE in decoder (prologue + ids wait) | decoder | +1% | small | after SMOE |
| SMOE M<=16 (two n8 mma per k-step) | moe, decoder | bs16 load | medium | idea |
| L2 persist window for MoE scratch | moe | <1% | small | idea |
Fresh-subagent ideas (2026-11-03, docs-only context): (1) byte-sized global ticket queue incl. K-slices for bs16-128; (2) PDL-launch the MoE early so its prologue/claims overlap the router (next after SMOE validates); (3) cluster-fused w13->w2 with DSMEM intermediate; (4) FP8 mma via e2m1->e4m3 (blocked: activations must stay BF16); (5) DRAM-channel-aware weight swizzle.

## Public-repo ideas, 2026-11-04 (from research/PUBLIC-CODE-REFRESH-2026-11-04.md; ideas only, no code copied)
| Idea | Credit / source | Board | Expected | Cost | Status |
|---|---|---|---|---|---|
| Split full/empty mbarriers into separate K and V readiness in the flat producer ring, so QK starts when K lands while V is still in flight | AutoclickerI/muse-spark-open-kernels (Apache-2.0), VQ8 page ring notes in docs/FINDINGS.md | attention full verify | +1-2% (their gain was small; transfer unproven) | medium | untried |
| Drop the redundant consumer wait their ring removed; check our `ws_acquire` (mbar_wait_full per tile) for the same redundancy | same | attention full | <1% | small | untried |
| Task-queue megakernel for the attention -> o-proj boundary (last-finishing CTAs take o-proj tiles) | idea from sgl-project/sgl-nvidia-mega (Apache-2.0, Blackwell-focused; H200 fit unknown) | decoder | +1-3% (one launch) | large | untried |
| Fixed-seed paired A/B protocol for every claim (judge nodes vary a few %; the 2.47x decoder post is unverified) | process lead from the Nov 4 user-supplied post | all | avoids false keeps | small | adopt |
| Packed-half exp2 lowers to two MUFU ops | AutoclickerI findings; matches our paired regression (NOTES.md) | attention | negative | none | do not retry |

## Public-repo ideas, 2026-11-04 evening (ideas only; credit listed)
| Idea | Credit / source | Board | Expected | Cost | Status |
|---|---|---|---|---|---|
| Move NVFP4 decode from the ALU pipe to the FMA pipe (their v26->v27 step: 1.678->1.804x) | Aravind-11/muse-moe-w4a16-kernels (no license file: idea only, no code) | moe, decoder | +2-5% if ALU-issue bound at bs16-256 | medium | assigned to moe_mid_v2 agent |
| Bare TMA-ring skeleton of the same bytes per shape as the ceiling (their measured MoE ceiling ~3.06x) | Aravind-11 README; keell0renz LESSONS.md section 5 | moe | sets target | small | moe_mid_v2 |
| MoE loss is events (3 us ramp, A->B dip, slow phase-B start at bs1, 6-9 us silent tail), not steady DRAM rate (ncu PM sampling) | keell0renz/muse-open-kernel-challenge LESSONS.md (Apache-2.0) | moe | targets the tail and ramp | n/a | relayed to agent |
| Router/top-k on a second stream overlapping the pooler in the decoder | ruwwww/open-kernel-challenge README (license unchecked: idea only) | decoder | +0.5-1% T>=16 | small | untried |
| Build N variants in untimed prepare() and time them back to back in one GPU session (10-40 experiments per Modal call) | keell0renz LESSONS.md | all | throughput of experiments | small | adopt in agents |
| Decoder T=128/256 projections: port the projection board's routes (chenwainuo/ag/nb, FP32 partials). The decoder's nvjet + ol_projection set takes ~70 us (verify16) / ~87 us (verify32), while the projection board does the same 4 layers in 60.3 / 74.5 us. | own data (decoder profile 2996c182, proj profile bucket_v7) | decoder verify16/32 | about 9-12 us per shape, about +1% decoder | medium (895 KB file; check epilogue fusions) | queued for the next free agent |
