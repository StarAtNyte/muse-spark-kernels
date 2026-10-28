# attention_fullverify_stages5_b32_v1

Job: `73979b25946b4792`
Correctness: True; official speedup: 1.86857x; CI: [1.8405368306918002, 1.9027164899597562]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.94 | 8.01 | 2.616x | 0.11 |
| decode 1 | full | 30.83 | 15.73 | 1.960x | 4.27 |
| decode 16 | sliding | 26.94 | 12.16 | 2.216x | 1.65 |
| decode 16 | full | 200.54 | 96.79 | 2.072x | 63.18 |
| decode 32 | sliding | 31.15 | 15.02 | 2.074x | 3.29 |
| decode 32 | full | 384.00 | 183.32 | 2.095x | 126.35 |
| verify 1 | sliding | 22.67 | 8.79 | 2.579x | 0.12 |
| verify 1 | full | 35.62 | 22.89 | 1.556x | 4.28 |
| verify 16 | sliding | 29.15 | 17.09 | 1.706x | 1.86 |
| verify 16 | full | 205.62 | 154.74 | 1.329x | 63.39 |
| verify 32 | sliding | 34.11 | 20.70 | 1.648x | 3.72 |
| verify 32 | full | 389.02 | 292.94 | 1.328x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.457 | 0.620 | 2.350x | 19.844x |
| decode 16 | 4.448 | 2.108 | 2.110x | 4.094x |
| decode 32 | 7.577 | 3.624 | 2.091x | 3.487x |
| verify 1 | 1.613 | 0.771 | 2.093x | 21.723x |
| verify 16 | 4.631 | 3.262 | 1.420x | 4.210x |
| verify 32 | 7.794 | 5.639 | 1.382x | 3.543x |

Recomputed geometric mean across loads: 1.868565x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  58.6%      1
_prepare_qkv                                                       1.6 µs  21.1%      1
_merge_norm_gate                                                   1.5 µs  20.3%      1
total                                                              7.4 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.4 µs  76.3%      1
_merge_norm_gate                                                   2.1 µs  15.5%      1
_prepare_qkv                                                       1.1 µs   8.2%      1
total                                                             13.7 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.3 µs  61.3%      1
_merge_norm_gate                                                   1.7 µs  20.1%      1
_prepare_qkv                                                       1.6 µs  18.6%      1
total                                                              8.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.5 µs  95.0%      1
_merge_norm_gate                                                   3.1 µs   3.2%      1
_prepare_qkv                                                       1.7 µs   1.8%      1
total                                                             96.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.3 µs  67.1%      1
_merge_norm_gate                                                   2.2 µs  17.6%      1
_prepare_qkv                                                       1.9 µs  15.3%      1
total                                                             12.4 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     179.1 µs  97.6%      1
_merge_norm_gate                                                   2.4 µs   1.3%      1
_prepare_qkv                                                       2.0 µs   1.1%      1
total                                                            183.5 µs
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
wg5_partials(unsigned char const*, unsigned char const*, ...      16.2 µs  80.2%      1
_merge_norm_gate                                                   2.8 µs  13.6%      1
_prepare_qkv                                                       1.2 µs   6.2%      1
total                                                             20.2 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   14.3 µs 100.0%      1
total                                                             14.3 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               156.7 µs 100.0%      1
total                                                            156.7 µs
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
_os__g_full_verify                                               293.2 µs 100.0%      1
total                                                            293.2 µs
```

