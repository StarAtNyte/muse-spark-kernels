# keell0renz Muse challenge snapshot and H200 screen (2026-11-02)

Source: [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge), commit `8e30e1be99`.

## Source comparison

The public BF16 projection, MoE W4A16, prefill, and whole decode layer source files are byte-identical to our current kernel entrypoints. Attention is the only substantive source delta. It combines a `cp.reduce.async.bulk` TMA merge, a 72-float padded workspace row for selected q_len=1 per-request kernels, and edits to cluster finalization/prologue behavior. Its README cautions that leaderboard highs can exceed same-box scored runs; use paired local measurements and official scored submissions as the authority.

The repository advertises a Nov 1 snapshot of decode #38, attention #35, BF16 #26, prefill #29, and MoE #58, based on roughly 460 experiments. Those results are useful as a research index, but most files are the same source we already timed. Our official scored runs of the unchanged sources were BF16 1.58528x (matches the 1.5853x incumbent), decode 2.24332x (below our 2.2603x incumbent), MoE 2.37884x (below our 2.3820x incumbent), and prefill 1.93214x (matches our 1.9321x incumbent). Their README itself reports lower judge runs than the best leaderboard rows for some files. Do not copy or resubmit those unchanged kernels. Their lossless six-bit WGS scale packing remains the relevant MoE delta; v8 reached the judge but failed before timing because its copied allocator omitted the MFIN scratch region. V9 restores the incumbent's scratch sizing and DF sentinel initialization; check `bd7ee2f9d89247c8` is queued.

Transfer summary: (1) their CUDA/measurement lessons are actionable and now captured in `research/ENGINEERING.md`; their cold-L2 flush, fresh inputs, and noisy-score observations reinforce our measurement discipline, (2) their new attention bulk-reduction/workspace changes were screened and showed no repeatable q_len=8 gain, with q_len=1 results inconclusive, and (3) their SC6 MoE method remains an unverified delta in our own timed path pending v9 check `bd7ee2f9d89247c8`. Nothing else in the current public snapshot is an untested source-level kernel delta over our baseline.

## H200 screens

Full upstream attention source, q_len=8 scored full-attention shapes, B16/B32, two seeds and 16 randomized cold-L2 pairs per seed:

| Batch | Seed 539 | Seed 1427 |
|---:|---:|---:|
| 16 | 1.0008× | 1.0008× |
| 32 | 0.9966× | 0.9999× |

Conclusion: no measurable q_len=8 gain. A broader q_len=1 screen showed small improvement signals only on sliding B16/B32; all other cases were effectively neutral. We then isolated removal of the dry cluster-finalizer pass. Its seed ratios were B16 1.0315×/0.9905× and B32 0.9965×/1.0112×, which is inconclusive. No upstream code or isolated change was promoted or officially submitted.

Modal logs and exact candidate sources are archived under `experiments/attention_keell0renz_bulk_v1/` and `experiments/attention_keell0renz_drycluster_remove_v1/`. The experiment scripts checked correctness against the task reference before timing. The source LICENSE and NOTICE are preserved alongside the upstream-derived candidate.

## Current judge queue observations

- MoE check `a5bdd94d9dc7460b` (v8) failed before timing on missing MFIN scratch storage; v9 check `bd7ee2f9d89247c8` is queued.
- Projection check `a7bd51ec7e13491a` finished but failed during Python parse at line 7241: injected base64 payload ran into `_KZ_SHAPES` (`SyntaxError`). This is a packaging/source-construction defect, not a kernel performance result; do not compare its score.

## Next useful experiment

Avoid retesting the unchanged upstream files. To follow up the only signal, port the smallest relevant q_len=1 workspace-layout/merge change onto our incumbent and screen only sliding B16/B32 with more than two seeds. Keep it isolated from finalizer/prologue edits. Only run the official check if the paired same-box gain is stable and the candidate passes reference checks.
