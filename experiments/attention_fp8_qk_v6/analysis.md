# attention_fp8_qk_v6

Job: `ca9e327f05c04b3d`
Correctness: True; official speedup: 1.17569x; CI: [1.1580538258512152, 1.1933245007502369]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.14 | 9.91 | 2.033x | 0.11 |
| decode 1 | full | 30.98 | 20.65 | 1.500x | 4.27 |
| decode 16 | sliding | 26.99 | 22.07 | 1.223x | 1.65 |
| decode 16 | full | 201.07 | 156.50 | 1.285x | 63.18 |
| decode 32 | sliding | 31.28 | 23.35 | 1.340x | 3.29 |
| decode 32 | full | 384.26 | 300.93 | 1.277x | 126.35 |
| verify 1 | sliding | 22.27 | 13.05 | 1.707x | 0.12 |
| verify 1 | full | 36.00 | 26.30 | 1.369x | 4.28 |
| verify 16 | sliding | 29.44 | 27.28 | 1.079x | 1.86 |
| verify 16 | full | 204.69 | 294.65 | 0.695x | 63.39 |
| verify 32 | sliding | 34.77 | 32.01 | 1.086x | 3.72 |
| verify 32 | full | 389.84 | 573.63 | 0.680x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.422 | 0.786 | 1.809x | 19.374x |
| decode 16 | 4.459 | 3.519 | 1.267x | 4.103x |
| decode 32 | 7.587 | 5.889 | 1.288x | 3.491x |
| verify 1 | 1.601 | 1.021 | 1.567x | 21.558x |
| verify 16 | 4.629 | 5.969 | 0.775x | 4.209x |
| verify 32 | 7.837 | 10.651 | 0.736x | 3.562x |

Recomputed geometric mean across loads: 1.175689x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  4.0 µs  56.8%      1
_prepare_qkv                                                       1.5 µs  21.6%      1
_merge_norm_gate                                                   1.5 µs  21.6%      1
total                                                              7.1 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 12.4 µs  79.3%      1
_merge_norm_gate                                                   2.1 µs  13.6%      1
_prepare_qkv                                                       1.1 µs   7.2%      1
total                                                             15.6 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 11.2 µs  78.7%      1
_prepare_qkv                                                       1.5 µs  10.8%      1
_merge_norm_gate                                                   1.5 µs  10.6%      1
total                                                             14.2 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                153.2 µs  97.4%      1
_merge_norm_gate                                                   2.2 µs   1.4%      1
_prepare_qkv                                                       2.0 µs   1.2%      1
total                                                            157.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 14.3 µs  80.5%      1
_prepare_qkv                                                       1.8 µs  10.3%      1
_merge_norm_gate                                                   1.6 µs   9.2%      1
total                                                             17.8 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                297.9 µs  98.6%      1
_merge_norm_gate                                                   2.4 µs   0.8%      1
_prepare_qkv                                                       2.0 µs   0.6%      1
total                                                            302.2 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.1 µs  66.2%      1
_merge_norm_gate                                                   1.7 µs  17.9%      1
_prepare_qkv                                                       1.5 µs  15.9%      1
total                                                              9.3 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 18.7 µs  82.6%      1
_merge_norm_gate                                                   2.7 µs  11.9%      1
_prepare_qkv                                                       1.2 µs   5.5%      1
total                                                             22.6 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 14.3 µs  74.6%      1
_merge_norm_gate                                                   3.0 µs  15.7%      1
_prepare_qkv                                                       1.9 µs   9.7%      1
total                                                             19.2 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                288.7 µs  97.6%      1
_merge_norm_gate                                                   5.0 µs   1.7%      1
_prepare_qkv                                                       2.2 µs   0.7%      1
total                                                            295.9 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 21.3 µs  73.6%      1
_merge_norm_gate                                                   4.9 µs  16.9%      1
_prepare_qkv                                                       2.7 µs   9.4%      1
total                                                             28.9 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                569.3 µs  98.5%      1
_merge_norm_gate                                                   6.7 µs   1.2%      1
_prepare_qkv                                                       2.3 µs   0.4%      1
total                                                            578.2 µs
```

