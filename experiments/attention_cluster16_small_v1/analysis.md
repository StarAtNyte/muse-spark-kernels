# attention_cluster16_small_v1

Job: `389d845575f84f5f`
Correctness: True; official speedup: 1.72676x; CI: [1.7008548056163115, 1.752657490051326]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.54 | 7.93 | 2.589x | 0.11 |
| decode 1 | full | 30.77 | 15.49 | 1.986x | 4.27 |
| decode 16 | sliding | 27.04 | 12.08 | 2.238x | 1.65 |
| decode 16 | full | 200.98 | 96.90 | 2.074x | 63.18 |
| decode 32 | sliding | 31.46 | 15.06 | 2.088x | 3.29 |
| decode 32 | full | 384.29 | 184.03 | 2.088x | 126.35 |
| verify 1 | sliding | 22.54 | 12.73 | 1.771x | 0.12 |
| verify 1 | full | 35.63 | 23.00 | 1.549x | 4.28 |
| verify 16 | sliding | 29.25 | 19.96 | 1.465x | 1.86 |
| verify 16 | full | 204.64 | 166.66 | 1.228x | 63.39 |
| verify 32 | sliding | 34.42 | 27.99 | 1.230x | 3.72 |
| verify 32 | full | 389.42 | 334.54 | 1.164x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.437 | 0.613 | 2.346x | 19.579x |
| decode 16 | 4.459 | 2.106 | 2.117x | 4.104x |
| decode 32 | 7.596 | 3.637 | 2.088x | 3.495x |
| verify 1 | 1.607 | 0.953 | 1.686x | 21.647x |
| verify 16 | 4.620 | 3.585 | 1.289x | 4.200x |
| verify 32 | 7.814 | 6.640 | 1.177x | 3.552x |

Recomputed geometric mean across loads: 1.726756x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  55.8%      1
_prepare_qkv                                                       2.0 µs  25.2%      1
_merge_norm_gate                                                   1.5 µs  19.0%      1
total                                                              7.7 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.2 µs  76.4%      1
_merge_norm_gate                                                   2.0 µs  15.4%      1
_prepare_qkv                                                       1.1 µs   8.2%      1
total                                                             13.3 µs
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
void trans_decode<3>(unsigned char const*, unsigned char ...      92.2 µs  94.6%      1
_merge_norm_gate                                                   3.4 µs   3.5%      1
_prepare_qkv                                                       1.8 µs   1.9%      1
total                                                             97.4 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.4 µs  69.9%      1
_merge_norm_gate                                                   1.8 µs  15.2%      1
_prepare_qkv                                                       1.8 µs  14.9%      1
total                                                             12.0 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     180.0 µs  97.6%      1
_merge_norm_gate                                                   2.5 µs   1.3%      1
_prepare_qkv                                                       1.9 µs   1.0%      1
total                                                            184.4 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...       8.6 µs  84.6%      1
_prepare_qkv                                                       1.6 µs  15.4%      1
total                                                             10.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      15.9 µs  80.8%      1
_merge_norm_gate                                                   2.6 µs  13.3%      1
_prepare_qkv                                                       1.2 µs   5.9%      1
total                                                             19.7 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      15.2 µs  88.9%      1
_prepare_qkv                                                       1.9 µs  11.1%      1
total                                                             17.1 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     160.7 µs  96.0%      1
_merge_norm_gate                                                   4.7 µs   2.8%      1
_prepare_qkv                                                       2.0 µs   1.2%      1
total                                                            167.4 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.8 µs  58.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.4 µs  17.6%      1
_prepare_qkv                                                       2.7 µs  10.5%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.0%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   6.0%      1
total                                                             25.2 µs
GPU busy (kernels overlap: PDL or streams)                        23.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     320.9 µs  95.7%      1
_merge_norm_gate                                                  11.9 µs   3.5%      1
_prepare_qkv                                                       2.5 µs   0.7%      1
total                                                            335.3 µs
```

