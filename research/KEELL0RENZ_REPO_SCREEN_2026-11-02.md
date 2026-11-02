# keell0renz Muse challenge snapshot and H200 screen (2026-11-02)

Source: [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge), commit `8e30e1be99`.

## Source comparison

The public BF16 projection, MoE W4A16, prefill, and whole decode layer source files are byte-identical to our current kernel entrypoints. Attention is the only substantive source delta. It combines a `cp.reduce.async.bulk` TMA merge, a 72-float padded workspace row for selected q_len=1 per-request kernels, and edits to cluster finalization/prologue behavior. Its README cautions that leaderboard highs can exceed same-box scored runs; use paired local measurements and official scored submissions as the authority.

## H200 screens

Full upstream attention source, q_len=8 scored full-attention shapes, B16/B32, two seeds and 16 randomized cold-L2 pairs per seed:

| Batch | Seed 539 | Seed 1427 |
|---:|---:|---:|
| 16 | 1.0008× | 1.0008× |
| 32 | 0.9966× | 0.9999× |

Conclusion: no measurable q_len=8 gain. A broader q_len=1 screen showed small improvement signals only on sliding B16/B32; all other cases were effectively neutral. We then isolated removal of the dry cluster-finalizer pass. Its seed ratios were B16 1.0315×/0.9905× and B32 0.9965×/1.0112×, which is inconclusive. No upstream code or isolated change was promoted or officially submitted.

Modal logs and exact candidate sources are archived under `experiments/attention_keell0renz_bulk_v1/` and `experiments/attention_keell0renz_drycluster_remove_v1/`. The experiment scripts checked correctness against the task reference before timing. The source LICENSE and NOTICE are preserved alongside the upstream-derived candidate.

## Current judge queue observations

- MoE check `d237a4892dc54420` remained queued at the last poll.
- Projection check `a7bd51ec7e13491a` finished but failed during Python parse at line 7241: injected base64 payload ran into `_KZ_SHAPES` (`SyntaxError`). This is a packaging/source-construction defect, not a kernel performance result; do not compare its score.

## Next useful experiment

Avoid retesting the unchanged upstream files. To follow up the only signal, port the smallest relevant q_len=1 workspace-layout/merge change onto our incumbent and screen only sliding B16/B32 with more than two seeds. Keep it isolated from finalizer/prologue edits. Only run the official check if the paired same-box gain is stable and the candidate passes reference checks.
