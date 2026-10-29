# Attention split-KV pipeline depth (non-flat Cfg STAGES 3 -> 4/5/6) — rejected

Driver `dev/modal/drv_attn_stages.py`, raw `modal.log` (CUDA 13 H200, paired cold-L2, reference-checked).
Only the STAGES constant changed. Gains vs scored (base) over two seeds:

| case | st4 | st5 | st6 |
|---|---:|---:|---:|
| decode 1 sliding | 0.97-0.98 | 0.98 | 1.00-1.01 |
| decode 1 full | 0.99 | 0.98 | 0.97-0.98 |
| verify 1 sliding | 1.00-1.02 | 1.00-1.01 | 1.00-1.01 |
| verify 1 full | 0.97-0.98 | 0.57 | 0.56 |
| decode 16 sliding | 0.99-1.00 | 0.99-1.01 | 0.99-1.00 |
| verify 16 sliding | 0.98 | 0.97 | 0.96 |
| decode 32 sliding | 0.98 | 0.96-0.97 | 0.97-0.98 |

Deeper per-CTA rings do not raise the single-request DRAM throughput (ncu: 1.65 TB/s, 12% occupancy at decode 1 full,
`experiments/attention_small_ncu_v1/ncu.log`); plan overrides (splits/coop/copies, `experiments/attention_plan_b1_v1`)
also found the scored plan optimal.
