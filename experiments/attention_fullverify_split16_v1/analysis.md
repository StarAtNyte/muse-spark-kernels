# attention_fullverify_split16_v1

Job: `8c7347b84b1b4e70`
Correctness: True; official speedup: 1.83339x; CI: [1.805888968832088, 1.8608906633142834]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.40 | 7.92 | 2.574x | 0.11 |
| decode 1 | full | 31.52 | 15.93 | 1.978x | 4.27 |
| decode 16 | sliding | 26.91 | 12.04 | 2.235x | 1.65 |
| decode 16 | full | 200.29 | 97.10 | 2.063x | 63.18 |
| decode 32 | sliding | 31.20 | 14.62 | 2.135x | 3.29 |
| decode 32 | full | 384.70 | 184.32 | 2.087x | 126.35 |
| verify 1 | sliding | 22.34 | 8.48 | 2.634x | 0.12 |
| verify 1 | full | 36.02 | 22.67 | 1.589x | 4.28 |
| verify 16 | sliding | 29.09 | 16.77 | 1.735x | 1.86 |
| verify 16 | full | 205.31 | 165.79 | 1.238x | 63.39 |
| verify 32 | sliding | 34.48 | 20.82 | 1.656x | 3.72 |
| verify 32 | full | 389.98 | 321.39 | 1.213x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.443 | 0.619 | 2.329x | 19.653x |
| decode 16 | 4.443 | 2.107 | 2.108x | 4.088x |
| decode 32 | 7.590 | 3.621 | 2.096x | 3.493x |
| verify 1 | 1.604 | 0.753 | 2.130x | 21.601x |
| verify 16 | 4.623 | 3.424 | 1.350x | 4.203x |
| verify 32 | 7.826 | 6.100 | 1.283x | 3.557x |

Recomputed geometric mean across loads: 1.833390x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  59.8%      1
_prepare_qkv                                                       1.5 µs  20.1%      1
_merge_norm_gate                                                   1.5 µs  20.1%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.4 µs  76.2%      1
_merge_norm_gate                                                   2.1 µs  15.8%      1
_prepare_qkv                                                       1.1 µs   8.0%      1
total                                                             13.6 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.2 µs  62.0%      1
_merge_norm_gate                                                   1.6 µs  19.4%      1
_prepare_qkv                                                       1.6 µs  18.6%      1
total                                                              8.4 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.7 µs  95.2%      1
_merge_norm_gate                                                   2.8 µs   3.0%      1
_prepare_qkv                                                       1.7 µs   1.8%      1
total                                                             96.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.2 µs  68.7%      1
_prepare_qkv                                                       1.9 µs  16.0%      1
_merge_norm_gate                                                   1.8 µs  15.2%      1
total                                                             12.0 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     179.3 µs  97.5%      1
_merge_norm_gate                                                   2.7 µs   1.5%      1
_prepare_qkv                                                       1.9 µs   1.0%      1
total                                                            183.9 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                    6.9 µs 100.0%      1
total                                                              6.9 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.4 µs  80.5%      1
_merge_norm_gate                                                   2.7 µs  13.1%      1
_prepare_qkv                                                       1.3 µs   6.5%      1
total                                                             20.3 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   14.7 µs 100.0%      1
total                                                             14.7 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               166.5 µs 100.0%      1
total                                                            166.5 µs
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
_os__g_full_verify                                               321.3 µs 100.0%      1
total                                                            321.3 µs
```

