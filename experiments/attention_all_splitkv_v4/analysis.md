# attention_all_splitkv_v4

Job: `eb6de29d0a314bef`
Correctness: True; official speedup: 1.01903x; CI: [1.003740064543757, 1.0343108279308766]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.02 | 10.32 | 1.940x | 0.11 |
| decode 1 | full | 31.01 | 22.19 | 1.398x | 4.27 |
| decode 16 | sliding | 27.10 | 22.99 | 1.179x | 1.65 |
| decode 16 | full | 201.36 | 185.79 | 1.084x | 63.18 |
| decode 32 | sliding | 31.76 | 25.16 | 1.262x | 3.29 |
| decode 32 | full | 384.59 | 357.52 | 1.076x | 126.35 |
| verify 1 | sliding | 22.37 | 14.27 | 1.567x | 0.12 |
| verify 1 | full | 36.11 | 31.17 | 1.159x | 4.28 |
| verify 16 | sliding | 29.02 | 29.66 | 0.979x | 1.86 |
| verify 16 | full | 205.20 | 364.15 | 0.564x | 63.39 |
| verify 32 | sliding | 34.53 | 34.74 | 0.994x | 3.72 |
| verify 32 | full | 390.13 | 718.62 | 0.543x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.417 | 0.830 | 1.708x | 19.301x |
| decode 16 | 4.469 | 4.030 | 1.109x | 4.112x |
| decode 32 | 7.614 | 6.877 | 1.107x | 3.504x |
| verify 1 | 1.607 | 1.155 | 1.391x | 21.641x |
| verify 16 | 4.618 | 7.191 | 0.642x | 4.199x |
| verify 32 | 7.830 | 13.096 | 0.598x | 3.559x |

Recomputed geometric mean across loads: 1.019025x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  4.3 µs  58.8%      1
_prepare_qkv                                                       1.5 µs  21.1%      1
_merge_norm_gate                                                   1.5 µs  20.2%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 14.2 µs  81.6%      1
_merge_norm_gate                                                   2.1 µs  12.2%      1
_prepare_qkv                                                       1.1 µs   6.3%      1
total                                                             17.4 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 12.0 µs  80.5%      1
_merge_norm_gate                                                   1.5 µs   9.9%      1
_prepare_qkv                                                       1.4 µs   9.6%      1
total                                                             14.9 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                180.7 µs  97.7%      1
_merge_norm_gate                                                   2.3 µs   1.3%      1
_prepare_qkv                                                       1.9 µs   1.0%      1
total                                                            184.9 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 15.6 µs  81.4%      1
_prepare_qkv                                                       1.8 µs   9.5%      1
_merge_norm_gate                                                   1.7 µs   9.0%      1
total                                                             19.1 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                353.7 µs  98.8%      1
_merge_norm_gate                                                   2.3 µs   0.6%      1
_prepare_qkv                                                       2.0 µs   0.6%      1
total                                                            358.0 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.9 µs  69.1%      1
_merge_norm_gate                                                   1.7 µs  16.6%      1
_prepare_qkv                                                       1.4 µs  14.3%      1
total                                                             10.0 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 23.3 µs  85.5%      1
_merge_norm_gate                                                   2.8 µs  10.1%      1
_prepare_qkv                                                       1.2 µs   4.3%      1
total                                                             27.2 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 16.3 µs  77.1%      1
_merge_norm_gate                                                   3.0 µs  14.1%      1
_prepare_qkv                                                       1.9 µs   8.8%      1
total                                                             21.1 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                356.5 µs  98.1%      1
_merge_norm_gate                                                   5.1 µs   1.4%      1
_prepare_qkv                                                       2.0 µs   0.5%      1
total                                                            363.6 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 23.9 µs  76.2%      1
_merge_norm_gate                                                   4.8 µs  15.2%      1
_prepare_qkv                                                       2.7 µs   8.6%      1
total                                                             31.4 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                711.6 µs  98.8%      1
_merge_norm_gate                                                   6.7 µs   0.9%      1
_prepare_qkv                                                       2.2 µs   0.3%      1
total                                                            720.5 µs
```

