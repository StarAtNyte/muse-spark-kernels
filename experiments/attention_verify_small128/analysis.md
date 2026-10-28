# attention_verify_small128

Job: `2191c82015cf4143`
Correctness: True; official speedup: 1.61142x; CI: [1.5872456820851002, 1.6355881901689102]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.59 | 7.93 | 2.595x | 0.11 |
| decode 1 | full | 31.41 | 15.91 | 1.974x | 4.27 |
| decode 16 | sliding | 27.07 | 13.52 | 2.002x | 1.65 |
| decode 16 | full | 201.38 | 122.05 | 1.650x | 63.18 |
| decode 32 | sliding | 31.66 | 17.22 | 1.839x | 3.29 |
| decode 32 | full | 383.18 | 207.61 | 1.846x | 126.35 |
| verify 1 | sliding | 22.82 | 13.50 | 1.690x | 0.12 |
| verify 1 | full | 36.34 | 22.86 | 1.589x | 4.28 |
| verify 16 | sliding | 29.41 | 22.80 | 1.290x | 1.86 |
| verify 16 | full | 206.16 | 174.50 | 1.181x | 63.39 |
| verify 32 | sliding | 34.69 | 28.23 | 1.229x | 3.72 |
| verify 32 | full | 390.18 | 338.08 | 1.154x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.450 | 0.620 | 2.340x | 19.749x |
| decode 16 | 4.467 | 2.575 | 1.735x | 4.111x |
| decode 32 | 7.587 | 4.114 | 1.844x | 3.491x |
| verify 1 | 1.631 | 0.987 | 1.653x | 21.967x |
| verify 16 | 4.651 | 3.841 | 1.211x | 4.229x |
| verify 32 | 7.838 | 6.708 | 1.169x | 3.563x |

Recomputed geometric mean across loads: 1.611417x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  59.1%      1
_prepare_qkv                                                       1.5 µs  20.4%      1
_merge_norm_gate                                                   1.5 µs  20.4%      1
total                                                              7.4 µs
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
void fd2_partials<3>(unsigned char const*, unsigned char ...       7.1 µs  70.5%      1
_prepare_qkv                                                       1.5 µs  14.9%      1
_merge_norm_gate                                                   1.5 µs  14.6%      1
total                                                             10.1 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     115.7 µs  94.8%      1
_merge_norm_gate                                                   4.4 µs   3.6%      1
_prepare_qkv                                                       1.9 µs   1.6%      1
total                                                            122.1 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...      11.1 µs  76.0%      1
_prepare_qkv                                                       1.9 µs  13.0%      1
_merge_norm_gate                                                   1.6 µs  11.0%      1
total                                                             14.6 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     202.7 µs  97.3%      1
_merge_norm_gate                                                   3.6 µs   1.8%      1
_prepare_qkv                                                       1.9 µs   0.9%      1
total                                                            208.3 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.0 µs  65.7%      1
_merge_norm_gate                                                   1.7 µs  18.5%      1
_prepare_qkv                                                       1.4 µs  15.7%      1
total                                                              9.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.3 µs  80.7%      1
_merge_norm_gate                                                   2.7 µs  13.3%      1
_prepare_qkv                                                       1.2 µs   6.0%      1
total                                                             20.2 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  54.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs  19.6%      1
_prepare_qkv                                                       2.0 µs  10.1%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.1%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.8%      1
total                                                             19.7 µs
GPU busy (kernels overlap: PDL or streams)                        18.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     167.3 µs  96.0%      1
_merge_norm_gate                                                   4.8 µs   2.8%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            174.2 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.7 µs  59.5%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs  17.2%      1
_prepare_qkv                                                       2.5 µs  10.2%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.7 µs   6.9%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   6.2%      1
total                                                             24.7 µs
GPU busy (kernels overlap: PDL or streams)                        23.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     325.1 µs  95.5%      1
_merge_norm_gate                                                  12.9 µs   3.8%      1
_prepare_qkv                                                       2.5 µs   0.7%      1
total                                                            340.4 µs
```

