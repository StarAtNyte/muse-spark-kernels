# claude_attention_rescale_lane_v1 findings

## Experiment
Replication of attention_fin_rescale_v1. `submission.py` is byte-identical to that experiment's (`cmp` equal). Versus parent `dev/modal/_attention_readable_parent.py` the only change is the finalization rescale: `pr8(x*inv)` and eps 1e-5 become `pr8(x)` and eps `1e-5*(lc/vscale)^2` (via `__fdividef`) when `1e-12 < inv < 1e12`; else original path.

Driver differs from the sibling only in: 6 trials (not 3), seeds 7300+bsz+q+211*trial, 24 reps (not 16) per median. Same 2 shapes: batch 1, ctx 80000, tp8, sliding, soft, page 1; decode q=1 and verify q=8. Other scored shapes not run.

## Results (modal.log, RETURNCODE 0, DONE printed)
Gain = ours/cand, >1 means candidate faster.

| trial | decode us ours/cand | gain | verify us ours/cand | gain |
|---|---|---|---|---|
| 0 | 8.448 / 8.576 | 0.98507 | 10.416 / 10.304 | 1.01087 |
| 1 | 8.592 / 8.768 | 0.97993 | 10.400 / 10.400 | 1.00000 |
| 2 | 8.560 / 8.576 | 0.99813 | 10.400 / 10.480 | 0.99237 |
| 3 | 8.512 / 8.544 | 0.99625 | 10.512 / 10.352 | 1.01546 |
| 4 | 8.576 / 8.608 | 0.99628 | 10.304 / 10.336 | 0.99690 |
| 5 | 8.512 / 8.672 | 0.98155 | 10.432 / 10.432 | 1.00000 |

Geomean (computed from printed gains): all 12 rows 0.99602; decode 0.98951; verify 1.00257.

## Correctness
All 12 rows `ok` true for ours and cand (`R.check_implementation`). Only the 2 shapes above; no okc judge run. `claude.log` is empty (1 byte), no extra evidence.

## Numerical / math risks
- Rescale identity RMS(x*s)=s*RMS(x) is exact only in real arithmetic; `pr8` now quantizes unscaled `x`, so FP8 rounding can differ near exponent/subnormal/saturation boundaries. Not tested beyond random soft-dist inputs.
- `__fdividef` is approximate; `vscale` is not bounded directly by the guard (only `inv`).

## Recommendation
Reject. Decode is slower in all 6 trials (0.980-0.998, geomean 0.9895); verify is noise (0.992-1.015, geomean 1.0026). Combined with the sibling's 3 trials (decode ~1.004, verify ~1.016), pooled 9 trials give no consistent gain, and 9 decode trials lean slightly negative. No evidence of benefit beyond noise.
