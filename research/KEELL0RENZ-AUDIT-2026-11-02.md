# Keell0renz public-kernel audit

Checked 2026-11-02 against [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge). The current public revision has two commits, both dated Nov 1: `8e30e1be` (README/open-source standing) and `3435290` (kernel release and lessons). The repo reports about 460 experiments and publishes H200 implementations for all five boards. Its README's Nov 1 ranks are an interesting public reference, but its own lessons document run-to-run noise and several high-draw leaderboard rows; use exact-code judge runs and paired H200 measurements to decide what to transfer.

## Exact-source judge runs

Four upstream source files were submitted unchanged and have completed official scoring. “Current” is our published incumbent from the Nov 2 standings snapshot. The confidence intervals overlap in every row, so small point differences do not establish that one implementation is faster.

| Board | Upstream exact-code score (95% CI) | Current incumbent | Job | Readout |
|---|---:|---:|---|---|
| Whole decoder | 2.24332× (2.20967–2.27697) | 2.2603× | `01bd05039098464e` | Lower point estimate; interval overlaps. |
| MoE W4A16 | 2.37884× (2.34315–2.41452) | 2.3820× | `43e6e35388774f73` | Near tie; no established gain. |
| BF16 projections | 1.58528× (1.56150–1.60905) | 1.5853× | `e5cae4f990be4df2` | Same score within noise. |
| Prefill layer | 1.93214× (1.90316–1.96113) | 1.9321× | `f3b10ea957e04ea6` | Same score within noise. |

The attention source passed a private full check at 2.61654× (95% CI 2.57729–2.65579), overlapping the 2.6328× incumbent. It has not been submitted for scoring. Its TMA bulk-reduction merge was also tested in paired Modal H200 runs: full-attention q_len=8 B16/B32 was neutral (0.9966–1.0008×); slight q_len=1 sliding signals were mixed across seeds. Removing the cluster dry pass did not produce a repeatable gain. These results and source/licensing records are in [`attention_keell0renz_compare_v1`](../experiments/attention_keell0renz_compare_v1/), [`attention_keell0renz_bulk_v1`](../experiments/attention_keell0renz_bulk_v1/), and [`attention_keell0renz_drycluster_remove_v1`](../experiments/attention_keell0renz_drycluster_remove_v1/).

The exact source archives and judge profiles are in:

- [`decode_keell0renz_compare_v1`](../experiments/scored_decode_keell0renz_compare_v1/)
- [`moe_keell0renz_compare_v1`](../experiments/scored_moe_keell0renz_compare_v1/)
- [`bf16_keell0renz_compare_v1`](../experiments/scored_bf16_keell0renz_compare_v1/)
- [`prefill_keell0renz_compare_v1`](../experiments/scored_prefill_keell0renz_compare_v1/)

## What transfers

The upstream implementations use several methods already present in our code: lossless 12-bit BF16 weight encoding and TMA-ring GEMVs, PDL chaining, per-batch-size persistent WGS MoE builds, warp-parallel stream-K scheduling, and shape-specific kernel dispatch. Exact-source comparisons found no reason to replace our current stack with a wholesale copy.

The reusable contribution is their measurement methodology: cold-L2 paired timing, same-codegen on/off controls, per-SM timelines, and explicit bare-ring ceilings before tuning. Their lessons also offer useful hypotheses for future targeted work: optimize the remaining MoE A→B handoff/tail and attention merge path only when our own profiles identify those intervals as critical. Do not infer speed from their headline board rows alone.
