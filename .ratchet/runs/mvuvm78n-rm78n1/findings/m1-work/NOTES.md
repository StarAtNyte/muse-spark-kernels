# m1: coordination, submission audit, next opportunity (2026-10-30)

## Outcome

No submission was made. No candidate in the workspace meets the promotion gate, which requires a
passing full official check whose result clearly beats the published score. No improvement is claimed.

## Candidate audit (evidence: audit.json and problems.json, plus manifests and status files)

| Candidate | Problem | Official score | Published | Verdict |
|---|---|---:|---:|---|
| attention_page_lookahead2_w2_v1 (75a9e2131a594077) | attention | 2.6018 [2.563, 2.657] | 2.6177 | reject; w2's paired Modal screen was also neutral |
| decoder_w4_gath7_plan16_v1 (56bffe6cfc6a4eae) | decoder | 2.1979 [2.165, 2.231] | 2.2603 | reject (w4) |
| moe_ours_kperm3_stack_v1 | MoE | 2.3911 | 2.3820 | **same sha256 as published (ac0081…)**: run-to-run noise, not a candidate |
| prefill_split_penalty20_v1 (2f13c4c92c8f4549) | prefill | 1.8040 | 1.7874 | dispatch is identical on scored shapes, so the gap is noise (decision already recorded in manifest) |
| prefill K64 fused (w5) | prefill | screened only | — | geomean 1.0019x with 2/4 regressions; not checked (correct) |

Noise calibration: rerunning identical code moved scores by +0.4% (MoE) and +0.9% (prefill).
A single official check cannot resolve gains below about 1%. Base gates on paired cold-L2 Modal screens.

Quota at audit time (from problems.json): proj 10 scored / 20 checks left; attention 10/19;
MoE 9/18; decoder 8/13; prefill 9/19.

## Worker state when this audit ran
- w1 MoE: L2-lookahead prefetch screens running on Modal (process 1448994). Left untouched.
- w2 attention: finished; negative result.
- w3 projections: running under antigravity (process 1439499); has not written findings yet. Left untouched.
- w4 decoder: finished; both candidates rejected; 2.2603x incumbent kept.
- w5 prefill: QKV-Lt judge-image screen running (process 1457217). Left untouched.
- w6 leaderboard intel: **died on the account's weekly limit**. The rank-leverage analysis below covers its main question.

## Rank leverage (lb_*.json snapshot from w6, 17:02)
Rank reached if the public score rose by the given percentage, with other teams unchanged:

| Board | now | +0.5% | +1% | +2% | +3% | +5% |
|---|---:|---:|---:|---:|---:|---:|
| projections | 147 | 112 | 87 | 33 | 21 | 5 |
| attention | 99 | 77 | 71 | 51 | 36 | 23 |
| MoE | 81 | 68 | 55 | 40 | 34 | 23 |
| decoder | 28 | 24 | 20 | 16 | 10 | 4 |

The overall board is "rank across the problems" (the API returns ranks only). If it aggregates
per-board ranks, **projections is by far the best place to spend effort**: +2% gains about 114 places.
Decoder is already near its plateau.

Observation: leaderboard rows carry `screened_speedup` and `screening: pending`. The leader shows
1.6498 published against 1.6292 screened. Public scores may be re-screened, so noise-lucky
resubmissions should not be relied on.

## Projection per-case gaps (published job beb3e26afb2544c9)
Every case already runs as one shape-specialized kernel. Large pooler and unpooler cases are at or
below SOL thanks to weight compression. The remaining headroom is in small, latency-bound cases.
"1 µs saved" is the gain in the geomean score:

- router: verify32 7.92 µs against SOL 2.68 (rwdl_256 kernel, 7.6 µs GPU); verify16 5.67/2.21; decode32 5.10/1.86;
  decode16 4.08/1.81; verify1 4.24/1.78; decode1 3.69/1.75. Each µs saved is worth 0.45–1.05%.
- unpooler m=1 and m=8: 3.75 and 4.52 µs against SOL 1.75. Each µs saved is worth about 0.8–1.0%.
- o: m=1/8/16/32 run 4.7–5.3 µs against SOL 3.5–3.6. Each µs saved is worth about 0.7–0.8%.
- o and qkv at m=128/256: 1.7–1.9x SOL.
The judge adds roughly 0.9 µs per case outside the kernel (timed minus profiled), so these cases sit
close to a launch-plus-DRAM-latency floor. The router at m=256 has the clearest kernel-side slack.

## Commands run (read-only, plus this notes file)
- Read README.md, the run's events.jsonl, claims.json, attention.md, decoder.md, worker findings, and research/best.json.
- Read experiments/*/manifest.json and status.json for the candidates above, plus tools/promote.py (the gate logic).
- Ran rank-leverage and per-case gap scripts (inline python) on findings/w6-work/lb_*.json and
  experiments/scored_projection_vjeux_hybrid_v1/profiles/submission.beb3e26afb2544c9.txt.
- Did not launch Modal jobs or official checks, to avoid contending with w1 and w5 for Modal H200
  and with w3 for projection quota.

## Next steps
1. Hand the projection gap list above to w3. Have w3 prioritize the router at m=256/128, then the
   small-m o and unpooler cases, before large-m GEMMs.
2. Restart w6's job, or drop it: the leverage table above answers where effort should go.
3. Attention has 108 archived experiments; obvious knobs (splits, grids, lookahead, cluster merge)
   are exhausted. Further gains need structural kernel work on the full decode/verify bandwidth
   efficiency (decode32 166 µs against 126 µs SOL; verify32 184 against 127).
4. Gate every promotion on a paired screen as well as an official check, given the ~1% noise between official runs.
