# Random 128 B KV row-gather bandwidth on H200 (2026-10-29)

Driver `dev/modal/drv_gather_bw.py`, raw `modal.log`. K and V caches [rows, 128 B] (2 KV heads x 64 FP8), gathered
through an index (random permutation = the judge's page=1 layout, or sequential). Cold L2, CUDA graph, median of 20;
empty graph 5.5 us.

| rows | order | best method | us | TB/s | TB/s minus launch |
|---:|---|---|---:|---:|---:|
| 2.56 M (655 MB) | random | LDG.128, 16 rows in flight per lane group | 180.0 | 3.64 | 3.76 |
| 2.56 M | sequential | same | 167.3 | 3.92 | 4.05 |
| 1.28 M (328 MB) | random | same | 94.6 | 3.46 | 3.68 |
| 1.28 M | sequential | same | 89.2 | 3.67 | 3.91 |

TMA bulk 128 B per row (ring depth 16/32/64) never reached the top five. In-flight depth beyond 2 rows barely
matters. The scored attention's decode-32 full case (~607 MB, 166.2 us) is at this random-gather limit, so the
attention leaders' ~18% margin cannot come from the large full-context cases; it must come from the latency-bound
sliding (weight 46) and small full cases.
