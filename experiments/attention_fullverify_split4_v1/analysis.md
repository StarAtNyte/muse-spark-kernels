# attention_fullverify_split4_v1

Job: `e3573ee956c84b92`
Correctness: True; official speedup: 1.79486x; CI: [1.767939838312043, 1.8217857217124098]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.70 | 7.91 | 2.616x | 0.11 |
| decode 1 | full | 31.78 | 16.05 | 1.980x | 4.27 |
| decode 16 | sliding | 26.62 | 11.68 | 2.279x | 1.65 |
| decode 16 | full | 200.21 | 96.72 | 2.070x | 63.18 |
| decode 32 | sliding | 31.28 | 14.66 | 2.134x | 3.29 |
| decode 32 | full | 384.03 | 184.20 | 2.085x | 126.35 |
| verify 1 | sliding | 22.85 | 9.10 | 2.509x | 0.12 |
| verify 1 | full | 36.05 | 22.53 | 1.600x | 4.28 |
| verify 16 | sliding | 29.25 | 16.90 | 1.731x | 1.86 |
| verify 16 | full | 205.30 | 212.16 | 0.968x | 63.39 |
| verify 32 | sliding | 34.85 | 21.31 | 1.636x | 3.72 |
| verify 32 | full | 389.60 | 294.14 | 1.325x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.461 | 0.621 | 2.353x | 19.899x |
| decode 16 | 4.428 | 2.085 | 2.124x | 4.075x |
| decode 32 | 7.583 | 3.621 | 2.094x | 3.489x |
| verify 1 | 1.628 | 0.779 | 2.089x | 21.925x |
| verify 16 | 4.630 | 4.172 | 1.110x | 4.209x |
| verify 32 | 7.837 | 5.686 | 1.378x | 3.562x |

Recomputed geometric mean across loads: 1.794863x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.4%      1
_prepare_qkv                                                       1.6 µs  21.2%      1
_merge_norm_gate                                                   1.5 µs  20.3%      1
total                                                              7.4 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.4 µs  76.6%      1
_merge_norm_gate                                                   2.1 µs  15.6%      1
_prepare_qkv                                                       1.1 µs   7.8%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.2 µs  61.4%      1
_merge_norm_gate                                                   1.7 µs  19.9%      1
_prepare_qkv                                                       1.6 µs  18.7%      1
total                                                              8.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.3 µs  95.0%      1
_merge_norm_gate                                                   2.9 µs   3.0%      1
_prepare_qkv                                                       1.9 µs   2.0%      1
total                                                             96.1 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.6 µs  69.8%      1
_prepare_qkv                                                       1.9 µs  15.2%      1
_merge_norm_gate                                                   1.9 µs  15.0%      1
total                                                             12.4 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     179.3 µs  97.6%      1
_merge_norm_gate                                                   2.5 µs   1.3%      1
_prepare_qkv                                                       2.0 µs   1.1%      1
total                                                            183.8 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                    7.1 µs 100.0%      1
total                                                              7.1 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      15.9 µs  79.9%      1
_merge_norm_gate                                                   2.7 µs  13.7%      1
_prepare_qkv                                                       1.3 µs   6.4%      1
total                                                             19.9 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   14.2 µs 100.0%      1
total                                                             14.2 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               212.0 µs 100.0%      1
total                                                            212.0 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   18.9 µs 100.0%      1
total                                                             18.9 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               292.4 µs 100.0%      1
total                                                            292.4 µs
```

