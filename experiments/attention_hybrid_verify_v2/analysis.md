# attention_hybrid_verify_v2

Job: `17f2722ab1f8495f`
Correctness: True; official speedup: 1.87981x; CI: [1.8516156926066507, 1.9080100791835028]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.82 | 7.88 | 2.643x | 0.11 |
| decode 1 | full | 31.38 | 15.75 | 1.992x | 4.27 |
| decode 16 | sliding | 27.14 | 12.36 | 2.196x | 1.65 |
| decode 16 | full | 201.02 | 96.99 | 2.073x | 63.18 |
| decode 32 | sliding | 31.01 | 14.59 | 2.126x | 3.29 |
| decode 32 | full | 382.75 | 181.85 | 2.105x | 126.35 |
| verify 1 | sliding | 22.34 | 8.11 | 2.753x | 0.12 |
| verify 1 | full | 36.42 | 22.46 | 1.621x | 4.28 |
| verify 16 | sliding | 28.66 | 16.27 | 1.761x | 1.86 |
| verify 16 | full | 205.63 | 155.49 | 1.322x | 63.39 |
| verify 32 | sliding | 34.43 | 20.90 | 1.647x | 3.72 |
| verify 32 | full | 390.51 | 306.18 | 1.275x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.460 | 0.614 | 2.376x | 19.882x |
| decode 16 | 4.465 | 2.120 | 2.106x | 4.109x |
| decode 32 | 7.550 | 3.581 | 2.109x | 3.474x |
| verify 1 | 1.610 | 0.733 | 2.198x | 21.687x |
| verify 16 | 4.608 | 3.236 | 1.424x | 4.190x |
| verify 32 | 7.832 | 5.860 | 1.336x | 3.560x |

Recomputed geometric mean across loads: 1.879813x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.2%      1
_prepare_qkv                                                       1.5 µs  20.9%      1
_merge_norm_gate                                                   1.5 µs  20.9%      1
total                                                              7.4 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.3 µs  76.5%      1
_merge_norm_gate                                                   2.1 µs  15.6%      1
_prepare_qkv                                                       1.1 µs   7.8%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.3 µs  61.3%      1
_merge_norm_gate                                                   1.8 µs  20.4%      1
_prepare_qkv                                                       1.6 µs  18.2%      1
total                                                              8.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      92.4 µs  95.1%      1
_merge_norm_gate                                                   3.0 µs   3.1%      1
_prepare_qkv                                                       1.8 µs   1.8%      1
total                                                             97.2 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.6 µs  70.3%      1
_merge_norm_gate                                                   1.9 µs  15.1%      1
_prepare_qkv                                                       1.8 µs  14.6%      1
total                                                             12.3 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     176.9 µs  97.4%      1
_merge_norm_gate                                                   2.6 µs   1.4%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            181.7 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                    7.2 µs 100.0%      1
total                                                              7.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      15.8 µs  79.6%      1
_merge_norm_gate                                                   2.7 µs  13.5%      1
_prepare_qkv                                                       1.4 µs   6.9%      1
total                                                             19.9 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   14.6 µs 100.0%      1
total                                                             14.6 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               154.3 µs 100.0%      1
total                                                            154.3 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   18.8 µs 100.0%      1
total                                                             18.8 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               309.3 µs 100.0%      1
total                                                            309.3 µs
```

