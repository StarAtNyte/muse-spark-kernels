# Exact-baseline batch-16 sliding attention-plan screen

## Decision
Reject for promotion. Candidate adds only `_TUNED7[(16, 1, True, 1)] = {n_splits: 8, coop: 0, nc: 1}` to the published decoder. Parent SHA-256 is `9e7c8b73b0471d04f4f0cf451bea6ad372d7356e3103e7f57d7253df234aebb2`; candidate SHA-256 is `5867686ff41d4378be716fb59364e2ca79dc2a8e0cb71959e1a14cd9ad90421e`.

Modal judge-image H200 paired screen covered sliding decode batches 1, 16 and 32 with two generated seeds each. All six pairs passed the reference check. Geometric speed ratios were 1.0001x, 0.9990x and 1.0008x respectively; overall paired geometric gain **1.0000x**. The target batch-16 ratios were 1.0022x and 0.9958x, so the apparent single-seed win does not repeat.

Raw output and exact invocation: `modal_judgeimage.log`. No official check or leaderboard submission was spent. Keep the 2.2603x decoder incumbent.
