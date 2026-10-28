# attention_transpose_selected

Job: `f75e704b61404709`
Correctness: True; official speedup: 1.70969x; CI: [1.684042791486261, 1.7353334348817815]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.00 | 7.60 | 2.633x | 0.11 |
| decode 1 | full | 31.52 | 16.08 | 1.960x | 4.27 |
| decode 16 | sliding | 26.69 | 11.89 | 2.245x | 1.65 |
| decode 16 | full | 200.83 | 96.81 | 2.074x | 63.18 |
| decode 32 | sliding | 31.46 | 14.96 | 2.103x | 3.29 |
| decode 32 | full | 384.74 | 183.14 | 2.101x | 126.35 |
| verify 1 | sliding | 22.96 | 13.48 | 1.703x | 0.12 |
| verify 1 | full | 36.27 | 22.37 | 1.622x | 4.28 |
| verify 16 | sliding | 29.22 | 22.40 | 1.305x | 1.86 |
| verify 16 | full | 205.09 | 172.67 | 1.188x | 63.39 |
| verify 32 | sliding | 34.70 | 28.18 | 1.232x | 3.72 |
| verify 32 | full | 390.30 | 337.57 | 1.156x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.424 | 0.607 | 2.348x | 19.402x |
| decode 16 | 4.441 | 2.096 | 2.119x | 4.087x |
| decode 32 | 7.603 | 3.618 | 2.101x | 3.498x |
| verify 1 | 1.637 | 0.978 | 1.673x | 22.043x |
| verify 16 | 4.625 | 3.793 | 1.219x | 4.205x |
| verify 32 | 7.841 | 6.697 | 1.171x | 3.564x |

Recomputed geometric mean across loads: 1.709688x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  56.2%      1
_prepare_qkv                                                       1.9 µs  24.4%      1
_merge_norm_gate                                                   1.5 µs  19.4%      1
total                                                              7.7 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.3 µs  76.2%      1
_merge_norm_gate                                                   2.1 µs  15.4%      1
_prepare_qkv                                                       1.1 µs   8.3%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.3 µs  61.8%      1
_merge_norm_gate                                                   1.7 µs  20.2%      1
_prepare_qkv                                                       1.5 µs  18.0%      1
total                                                              8.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      92.0 µs  95.0%      1
_merge_norm_gate                                                   3.0 µs   3.1%      1
_prepare_qkv                                                       1.8 µs   1.9%      1
total                                                             96.8 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.4 µs  70.3%      1
_merge_norm_gate                                                   1.8 µs  15.2%      1
_prepare_qkv                                                       1.7 µs  14.4%      1
total                                                             12.0 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     178.4 µs  97.6%      1
_merge_norm_gate                                                   2.6 µs   1.4%      1
_prepare_qkv                                                       1.8 µs   1.0%      1
total                                                            182.8 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.1 µs  66.4%      1
_merge_norm_gate                                                   1.6 µs  17.8%      1
_prepare_qkv                                                       1.4 µs  15.7%      1
total                                                              9.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.0 µs  80.6%      1
_merge_norm_gate                                                   2.7 µs  13.5%      1
_prepare_qkv                                                       1.2 µs   5.8%      1
total                                                             19.8 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.8 µs  54.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs  20.2%      1
_prepare_qkv                                                       2.0 µs   9.9%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.1%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.4 µs   6.8%      1
total                                                             20.1 µs
GPU busy (kernels overlap: PDL or streams)                        18.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     165.7 µs  96.0%      1
_merge_norm_gate                                                   4.8 µs   2.8%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            172.6 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.6 µs  60.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.0 µs  16.6%      1
_prepare_qkv                                                       2.5 µs  10.3%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.7 µs   7.1%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   6.0%      1
total                                                             24.4 µs
GPU busy (kernels overlap: PDL or streams)                        22.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     324.8 µs  95.5%      1
_merge_norm_gate                                                  12.7 µs   3.7%      1
_prepare_qkv                                                       2.4 µs   0.7%      1
total                                                            340.0 µs
```

