# attention_operatorstack_v1

Job: `af2ead39acc24eba`
Correctness: True; official speedup: 1.78669x; CI: [1.7598899338138003, 1.819090304690895]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.24 | 8.81 | 2.297x | 0.11 |
| decode 1 | full | 31.49 | 17.63 | 1.786x | 4.27 |
| decode 16 | sliding | 26.66 | 12.23 | 2.180x | 1.65 |
| decode 16 | full | 200.85 | 96.39 | 2.084x | 63.18 |
| decode 32 | sliding | 31.34 | 15.81 | 1.983x | 3.29 |
| decode 32 | full | 385.15 | 183.89 | 2.095x | 126.35 |
| verify 1 | sliding | 23.15 | 8.97 | 2.580x | 0.12 |
| verify 1 | full | 36.35 | 29.02 | 1.253x | 4.28 |
| verify 16 | sliding | 29.22 | 16.74 | 1.745x | 1.86 |
| verify 16 | full | 205.18 | 154.38 | 1.329x | 63.39 |
| verify 32 | sliding | 34.58 | 21.42 | 1.614x | 3.72 |
| verify 32 | full | 389.78 | 306.25 | 1.273x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.435 | 0.688 | 2.087x | 19.546x |
| decode 16 | 4.440 | 2.105 | 2.110x | 4.086x |
| decode 32 | 7.604 | 3.669 | 2.072x | 3.499x |
| verify 1 | 1.647 | 0.877 | 1.877x | 22.179x |
| verify 16 | 4.627 | 3.240 | 1.428x | 4.206x |
| verify 32 | 7.827 | 5.885 | 1.330x | 3.558x |

Recomputed geometric mean across loads: 1.786690x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                        7.1 µs 100.0%      1
total                                                              7.1 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                       14.8 µs 100.0%      1
total                                                             14.8 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                        9.4 µs 100.0%      1
total                                                              9.4 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_g_full_decode                                                    95.6 µs 100.0%      1
total                                                             95.6 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                       11.6 µs 100.0%      1
total                                                             11.6 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_g_full_decode                                                   185.4 µs 100.0%      1
total                                                            185.4 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                        6.9 µs 100.0%      1
total                                                              6.9 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                       26.1 µs 100.0%      1
total                                                             26.1 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                       14.6 µs 100.0%      1
total                                                             14.6 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_g_full_verify                                                   153.5 µs 100.0%      1
total                                                            153.5 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_fused_attn                                                       18.8 µs 100.0%      1
total                                                             18.8 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_g_full_verify                                                   304.3 µs 100.0%      1
total                                                            304.3 µs
```

