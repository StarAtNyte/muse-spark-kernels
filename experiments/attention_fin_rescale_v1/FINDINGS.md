# attention_fin_rescale_v1 findings

## Experiment
Candidate `submission.py` vs parent `dev/modal/_attention_readable_parent.py`. Diff is one hunk (~line 905-917) in the finalization step. Parent computes `pr8(x*inv)` then `rsqrtf(ss/64 + 1e-5)`. Candidate, when `inv` is in (1e-12, 1e12), uses `pr8(x*1)` and `rsqrtf(ss/64 + 1e-5*(lc/vscale)^2)`, using `__fdividef`. Idea: RMS(x*s) = s*RMS(x), so move the scale into epsilon and skip the multiply. Outside the range it falls back to the original path.

Driver: `driver.py`, paired alternating cold-L2 (256 MiB flush) CUDA-graph replays, median of 16 reps per trial, 3 trials (seeds 5000+bsz+q+101*trial). Only 2 shapes: batch 1, ctx 80000, tp8, sliding, soft, page 1; decode (q=1) and verify (q=8). The other scored shapes were not run.

## Results (modal.log, RETURNCODE 0, DONE printed)
Gain = ours/cand, >1 means candidate faster.

| trial | decode us ours/cand | gain | verify us ours/cand | gain |
|---|---|---|---|---|
| 0 | 8.784 / 8.864 | 0.99097 | 10.816 / 10.688 | 1.01198 |
| 1 | 8.992 / 8.992 | 1.00000 | 10.816 / 10.624 | 1.01807 |
| 2 | 8.976 / 8.784 | 1.02186 | 10.912 / 10.720 | 1.01791 |

Geomean (computed from the printed gains): all 6 rows 1.01007; decode 1.00419; verify 1.01598.

## Correctness
All 6 rows: `ok` true for both ours and cand via `R.check_implementation`. Covers only the 2 shapes above, no other shapes, no judge (okc) run.

## Numerical / math risks
- Equivalence of the epsilon rescale is exact in real arithmetic only. In float, `pr8` now rounds `x` instead of `x*inv`; if `pr8` is FP8 rounding, quantization differs whenever `inv` shifts values across FP8 exponent/subnormal/saturation boundaries. Not examined here; passed checks are only on random soft-dist inputs.
- `__fdividef` is approximate and returns 0/garbage for large denominators; `vscale` range is guarded only indirectly through the `inv` window (not verified that vscale itself is bounded).
- Gain ~1% is at the level of the judge noise (task notes: a few %).

## Recommendation
Not promote on this evidence. Decode gain is mixed (0.991, 1.000, 1.022), verify is consistently ~+1.2-1.8% but tiny and from 3 trials. The sibling experiment (identical source, 6 trials, different seeds) did not reproduce it (see claude_attention_rescale_lane_v1/FINDINGS.md). Reject unless a full 12-shape paired run or okc check shows gain beyond CI.
