# Next step: structural MoE streaming rewrite (brief for a fresh session)

State 2026-11-01: overall rank #61, set by the decoder score (2.2603x public, top-50 cutoff 2.279x). MoE 2.382x (#150).
KPERM3 (alygg77, Apache-2.0) is already ported in our own MoE and decoder; public repos have nothing further to stack
(attention/prefill there are vjeux's unchanged files; Kushagrabainsla and trxphxle are far behind us).

## Why a rewrite and not more tuning
- WGS MoE streams at ~3.3 TB/s vs ~4.1 TB/s contiguous-TMA floor (`research/FLOORS-2026-10-29.md`).
- Per-unit ends, not ring depth, bound the stream: every 16-stage unit end stalls a CTA 1-7 us; the slowest CTA moves to
  another range when tails are rebalanced (alygg77 NEGATIVE_RESULTS MoE #2; our `moe_wgs_stalls_v1`).
- Decode is already 7 ALU + 2 IMAD per 8 weights; removing the whole decode gains only 2-5% (math ablation). Do not
  spend time on dq4 micro-ops.
- Measured null: MID vs WGS at 32 tokens on the judge seed 3357109366520139634 (1.0005x), grid < 132 CTAs (0.91-0.99x),
  GATHB 7 on KPERM3 (+0.1-0.6%), 4-stage overlay, L2 prefetch, split-K single-token, FT_T=32 (ptxas regs).

## Weakest decoder case
batch=32 sliding decode: 148 us vs sglang 271 (1.83x), MoE 92 us = 54% of it (profile 4 in
`experiments/scored_decoder_compact_qkv_o_v1/profiles/submission.b72f02c7eb8745f5.txt`). The 32-token decode path is the
best target: byte floor is ~28 MB per touched expert at 4.1 TB/s.

## Candidate designs (untried)
1. Stream-K at expert-chunk granularity with a persistent producer warpgroup that never waits on unit ends (units
   fixed up through the existing WGS fixup slots, 28 free) so tails overlap the next expert's loads.
2. ALREADY IMPLEMENTED, do not redo: the WGS kernel is stream-K over 256-row units x 16 stages with cost-weighted
   stage ranges (`wgs_range`, `wgs_split`, `S.wgr`, cost tables OKC_WGSA/OKC_WGSB) and fix-up slots. What remains is the
   phase A (gate/up) -> phase B (down) dependency and the phase-B end spread (3.5 us at d32). A real gain needs those
   two phases overlapped across experts (phase B of expert i while phase A of expert i+1 streams), which means a
   per-expert dependency counter instead of a global phase barrier.

## Rules for the experiment
- Judge noise is ~1.5% on the decoder; trust only paired Modal screens with the judge's per-case seeds (status.json
  `result.records[].case.seed`) and >3% gains, then one `okc check`.
- Identical source scores as "duplicate"; do not resubmit for a lucky draw: the organizers rank on medians of reruns.
- Base: `experiments/moe_ours_kperm3_stack_v1/submission.py` (2.391x check). Modal harness:
  `dev/modal/run_judge_image.py` with `experiments/moe_mid_judgeseed_v1/driver.py` as the paired template.
