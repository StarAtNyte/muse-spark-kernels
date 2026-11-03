# Attention score leverage from the current public run

Source: official scored job `7d250883e6b94c50`, archived in `status.json`.
The current score is 2.6328x (#163); current #1 is 3.3389x. Attention's
objective is the geometric mean of six load speedups, with each load's latency
formed from its weighted full and sliding cases.

`python tools/prioritize.py --problem muse_attention_decode --status status.json`
reports the following load shares. The last column is a conditional score
sensitivity if that one case's latency alone drops 20%; it is not a prediction
that the reduction is attainable.

| Load | Case | Latency | Share of load | Conditional score gain |
|---|---|---:|---:|---:|
| Decode 32 | Full | 165.94 us | 85.2% | 3.16% |
| Verify 32 | Full | 184.86 us | 81.3% | 3.00% |
| Decode 16 | Full | 86.80 us | 77.6% | 2.85% |
| Verify 16 | Full | 98.98 us | 75.3% | 2.76% |

These four cases are the main score levers. The current B32 full-verify NCU
capture reads 669.45 MB and reaches 3.61 TB/s; its copy-load sector count is
consistent with efficient 16-sector requests. The 3.6--3.76 TB/s gather
comparator makes a simple coalescing rewrite a weak hypothesis. The 2- and
3-stage rings were slower than four stages; five stages were statistically
unchanged and slightly slower in this profile. See
[`attention_current_verify32_ncu_v1`](../attention_current_verify32_ncu_v1/analysis.md)
and the stage experiments for the raw evidence.

Next attention work should target greater HBM request concurrency or reduce
non-overlapped kernel work while preserving the required FP8 key/value traffic
and attending every key. An occupancy-only change or a page-size-1 TMA swap
has no supporting evidence yet. Any new mapping must beat this exact scored
source in a paired H200 screen and still pass the complete official check.
