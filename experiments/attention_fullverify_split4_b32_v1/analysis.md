# attention_fullverify_split4_b32_v1

Job: `768aaec1b20d430a`
Correctness: True; official speedup: 1.88401x; CI: [1.855750070260548, 1.9122703769689908]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.22 | 7.76 | 2.606x | 0.11 |
| decode 1 | full | 30.53 | 15.50 | 1.970x | 4.27 |
| decode 16 | sliding | 26.85 | 11.99 | 2.238x | 1.65 |
| decode 16 | full | 199.23 | 96.10 | 2.073x | 63.18 |
| decode 32 | sliding | 31.15 | 14.61 | 2.132x | 3.29 |
| decode 32 | full | 382.64 | 182.85 | 2.093x | 126.35 |
| verify 1 | sliding | 22.46 | 8.14 | 2.761x | 0.12 |
| verify 1 | full | 35.28 | 22.61 | 1.561x | 4.28 |
| verify 16 | sliding | 28.96 | 16.57 | 1.748x | 1.86 |
| verify 16 | full | 203.71 | 153.73 | 1.325x | 63.39 |
| verify 32 | sliding | 34.32 | 20.69 | 1.658x | 3.72 |
| verify 32 | full | 388.82 | 292.11 | 1.331x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.419 | 0.605 | 2.345x | 19.326x |
| decode 16 | 4.423 | 2.089 | 2.117x | 4.070x |
| decode 32 | 7.555 | 3.598 | 2.100x | 3.477x |
| verify 1 | 1.598 | 0.736 | 2.171x | 21.521x |
| verify 16 | 4.592 | 3.222 | 1.425x | 4.174x |
| verify 32 | 7.800 | 5.626 | 1.386x | 3.546x |

Recomputed geometric mean across loads: 1.884010x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.8%      1
_prepare_qkv                                                       1.5 µs  20.6%      1
_merge_norm_gate                                                   1.5 µs  20.6%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.3 µs  76.2%      1
_merge_norm_gate                                                   2.0 µs  15.2%      1
_prepare_qkv                                                       1.2 µs   8.6%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.2 µs  61.0%      1
_merge_norm_gate                                                   1.7 µs  20.1%      1
_prepare_qkv                                                       1.6 µs  19.0%      1
total                                                              8.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      90.7 µs  95.0%      1
_merge_norm_gate                                                   2.9 µs   3.0%      1
_prepare_qkv                                                       1.9 µs   2.0%      1
total                                                             95.5 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.5 µs  69.5%      1
_prepare_qkv                                                       1.9 µs  15.6%      1
_merge_norm_gate                                                   1.8 µs  14.8%      1
total                                                             12.3 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     179.0 µs  97.6%      1
_merge_norm_gate                                                   2.7 µs   1.4%      1
_prepare_qkv                                                       1.7 µs   0.9%      1
total                                                            183.4 µs
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
wg5_partials(unsigned char const*, unsigned char const*, ...      16.1 µs  80.4%      1
_merge_norm_gate                                                   2.7 µs  13.4%      1
_prepare_qkv                                                       1.2 µs   6.2%      1
total                                                             20.0 µs
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
_os__g_full_verify                                               154.8 µs 100.0%      1
total                                                            154.8 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   18.7 µs 100.0%      1
total                                                             18.7 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               292.3 µs 100.0%      1
total                                                            292.3 µs
```

