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
2. ALREADY IMPLEMENTED, do not redo: stream-K over 256-row units x 16 stages with cost-weighted stage ranges
   (`wgs_range`, `wgs_split`, `S.wgr`, OKC_WGSA/OKC_WGSB) and fix-up slots.
3. ALSO ALREADY IMPLEMENTED: phase A -> phase B overlap. Phase B waits per (job, stage) on `p.doneAs[j*16+st] >= 8`
   (`ld_acquire` spin, line ~2012) and phase A publishes row readiness per job in `p.xready[j]`; there is no global
   phase barrier. So design 1 above is the only structural lever left, and it competes with the same unit-end
   stalls already measured (alygg77 negative result #2: sub-unit claims need fix-ups whose tails eat the gain).
   Expected ceiling is small (MoE gap to the TMA floor is 13-33%, but every tail/overlap variant tried so far landed
   inside judge noise). Treat the MoE kernel as structurally near its limit before investing days.

## Rules for the experiment
- Judge noise is ~1.5% on the decoder; trust only paired Modal screens with the judge's per-case seeds (status.json
  `result.records[].case.seed`) and >3% gains, then one `okc check`.
- Identical source scores as "duplicate"; do not resubmit for a lucky draw: the organizers rank on medians of reruns.
- Base: `experiments/moe_ours_kperm3_stack_v1/submission.py` (2.391x check). Modal harness:
  `dev/modal/run_judge_image.py` with `experiments/moe_mid_judgeseed_v1/driver.py` as the paired template.

## Ceiling estimate (2026-11-01, from experiments/moe_timeline_v1)
At bs=32 decode (126 us kernel) CTA finish times spread 78.3-86.5 us (phase A) and 116.2-124.9 us (phase B), then +1 us
barrier and +3 us final. Perfect balancing would save at most ~8 us of 126 (6% on that one load; ~1-1.5% on the MoE board
score, since only d16/d32/v-loads carry such spreads). alygg77 measured that dynamic claims / sub-unit splits move the
last finisher later (unit 18.5 us vs 3.5 us spread; fix-up tails 1.5-4 us). Expect the rewrite to deliver +0-1.5% on the
MoE board and well under +1% on the decoder, i.e. inside judge noise. Do not start it expecting a rank change.

## Attention headroom map (2026-11-01, experiments/attention_ours_recheck_v1 profile)
Full-context decode is at the random-gather limit (decode 32 full: ~655 MB in 166 us = ~3.9 TB/s). Remaining gaps: full
verify (16 req: 101 us vs 87 us for same-bytes decode; 32 req: 185 vs 166) = q_len-8 compute on top of the gather, worth
~+3-4% on those two cases if half is closed; sliding layers are 5-15 us of mostly fixed launch/ramp cost. Prior attempts at
the verify-full gap (our wgmma wg1-wg5: parity with FA3; keell0renz A12 full-verify 2-head CTAs: no gain) were dead ends.
The reference plants a shared direction in low-attention keys' values, so token-skipping approximations fail the check.
