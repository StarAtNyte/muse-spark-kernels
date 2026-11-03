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
