Candidate: attention_readable_fp32_path_v1; official private check 12167249d81c46ee
Assumes other cases fixed; independent per-shape improvement, not a measured speedup.
| Load | Layer | Time µs | Lower bound µs | Score gain for 1 µs saved | Score gain for 10% case speedup |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 4.79 | 0.11 | 2.116% | 0.883% |
| decode 1 | full | 10.59 | 4.27 | 0.701% | 0.674% |
| decode 16 | sliding | 8.28 | 1.65 | 0.438% | 0.328% |
| decode 16 | full | 87.39 | 63.18 | 0.151% | 1.243% |
| decode 32 | sliding | 10.01 | 3.29 | 0.245% | 0.223% |
| decode 32 | full | 168.17 | 126.35 | 0.085% | 1.356% |
| verify 1 | sliding | 6.53 | 0.12 | 1.527% | 0.887% |
| verify 1 | full | 14.29 | 4.28 | 0.513% | 0.670% |
| verify 16 | sliding | 11.25 | 1.86 | 0.362% | 0.371% |
| verify 16 | full | 101.58 | 63.39 | 0.125% | 1.198% |
| verify 32 | sliding | 14.90 | 3.72 | 0.211% | 0.287% |
| verify 32 | full | 185.65 | 126.78 | 0.073% | 1.287% |

The 1 µs column is a mathematical sensitivity, not an estimate that this much can actually be saved. The lower bound is the judge's ideal bandwidth/compute estimate, not a demonstrated attainable latency; launch overhead, paging and scheduling can raise the practical floor. Larger full-context cases benefit most from a proportional improvement, while a small absolute batch-one sliding win has disproportionate board impact because it repeats 46 times per load.
