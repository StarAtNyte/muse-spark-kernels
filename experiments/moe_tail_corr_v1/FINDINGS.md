# WGS MoE phase tails are systematic (2026-10-29)

Per-CTA finish stamps over repeated runs of one input (`dev/modal/drv_moe_tail_corr.py`):

| bs | phase A spread | A run-to-run corr | phase B spread | B corr |
|---:|---:|---:|---:|---:|
| 32 | 8.3 us | 0.55 | 7.3 us | 0.46 |
| 128 | 18.1 us | 0.92 | 10.7 us | 0.67 |
| 256 | 14.6 us | 0.92 | 10.8 us | 0.70 |

Features (`experiments/moe_tail_features_v1`, `dev/modal/drv_moe_tail_features.py`, exported stage ranges):
phase-A finish correlates with the CTA's stage count (0.88/0.91) and anti-correlates with its jobs' n-tile count
(-0.82/-0.88): small-job stages are under-costed in WGSA. Phase B at bs128 correlates with n-tiles (+0.57):
large-job stages are under-costed in WGSB. Scaling whole tables (earlier sweep) is shape-invariant and cannot help;
flattening both tables together (`experiments/moe_cost_shape_v1/modal.log`) hurts badly (0.76-0.98x) because B
needs the opposite change. Independent A/B shape sweep in progress (`ab.log`).
