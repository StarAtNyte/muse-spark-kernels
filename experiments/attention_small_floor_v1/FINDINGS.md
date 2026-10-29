# Attention latency-bound cases vs gather floor (2026-10-29)

Same harness for both (cold L2, CUDA graph, event-timed median incl. ~5.5 us empty-graph cost).
Scored attention: `dev/modal/drv_attn_small.py` (CUDA 13 H200, log here). Floors: `dev/modal/drv_gather_bw.py`
with ROWS=80000,2048,32768 (`experiments/h200_gather_bw_v1/small.log`): index load + random 128 B K and V rows.

| case | scored attention us | pure gather floor us | gap |
|---|---:|---:|---:|
| decode 1 sliding (2,048 keys) | 8.99 | 7.0 | 2.0 |
| decode 1 full (80K keys, 20 MB) | 16.62 | 13.2 | 3.4 |
| verify 1 sliding | 11.07 | ~7.0 | ~4 |
| verify 1 full | 19.38 | 13.2 | 6.2 |

Attention needs ~2 more L2 round trips than a pure gather (query load overlaps; cross-CTA merge), so a realistic
target is gather floor + ~1 us. These shapes carry the decode-1 / verify-1 loads (sliding weight 46): halving the
gaps would add ~6% to the attention score.
