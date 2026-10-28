# attention_ptx_merge_v1

Job: `ac952a38225f454c`
Correctness: True; official speedup: 1.57311x; CI: [1.5495150412778145, 1.5967083927888137]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.53 | 7.82 | 2.626x | 0.11 |
| decode 1 | full | 31.31 | 15.60 | 2.008x | 4.27 |
| decode 16 | sliding | 26.96 | 13.50 | 1.996x | 1.65 |
| decode 16 | full | 200.56 | 121.82 | 1.646x | 63.18 |
| decode 32 | sliding | 31.15 | 17.14 | 1.818x | 3.29 |
| decode 32 | full | 383.78 | 208.15 | 1.844x | 126.35 |
| verify 1 | sliding | 22.70 | 13.40 | 1.694x | 0.12 |
| verify 1 | full | 36.42 | 26.68 | 1.365x | 4.28 |
| verify 16 | sliding | 28.99 | 22.38 | 1.295x | 1.86 |
| verify 16 | full | 205.25 | 185.45 | 1.107x | 63.39 |
| verify 32 | sliding | 34.46 | 28.08 | 1.227x | 3.72 |
| verify 32 | full | 389.25 | 357.76 | 1.088x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.445 | 0.609 | 2.373x | 19.688x |
| decode 16 | 4.449 | 2.570 | 1.731x | 4.094x |
| decode 32 | 7.573 | 4.119 | 1.839x | 3.485x |
| verify 1 | 1.627 | 1.043 | 1.560x | 21.915x |
| verify 16 | 4.618 | 3.997 | 1.155x | 4.198x |
| verify 32 | 7.813 | 7.016 | 1.114x | 3.552x |

Recomputed geometric mean across loads: 1.573112x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.5%      1
_prepare_qkv                                                       1.5 µs  21.0%      1
_merge_norm_gate                                                   1.5 µs  20.5%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.3 µs  76.3%      1
_merge_norm_gate                                                   2.1 µs  15.9%      1
_prepare_qkv                                                       1.1 µs   7.8%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       7.1 µs  71.1%      1
_prepare_qkv                                                       1.4 µs  14.5%      1
_merge_norm_gate                                                   1.4 µs  14.5%      1
total                                                             10.0 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     114.4 µs  94.8%      1
_merge_norm_gate                                                   4.5 µs   3.7%      1
_prepare_qkv                                                       1.9 µs   1.5%      1
total                                                            120.7 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...      10.9 µs  76.3%      1
_prepare_qkv                                                       1.8 µs  12.5%      1
_merge_norm_gate                                                   1.6 µs  11.2%      1
total                                                             14.3 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     202.9 µs  97.4%      1
_merge_norm_gate                                                   3.5 µs   1.7%      1
_prepare_qkv                                                       2.0 µs   1.0%      1
total                                                            208.4 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.0 µs  66.8%      1
_merge_norm_gate                                                   1.6 µs  18.0%      1
_prepare_qkv                                                       1.4 µs  15.2%      1
total                                                              9.1 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 18.5 µs  82.7%      1
_merge_norm_gate                                                   2.7 µs  12.0%      1
_prepare_qkv                                                       1.2 µs   5.3%      1
total                                                             22.4 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  54.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.0 µs  20.2%      1
_prepare_qkv                                                       1.9 µs   9.7%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.4 µs   6.9%      1
total                                                             19.8 µs
GPU busy (kernels overlap: PDL or streams)                        18.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     179.2 µs  96.3%      1
_merge_norm_gate                                                   4.8 µs   2.6%      1
_prepare_qkv                                                       2.0 µs   1.1%      1
total                                                            186.0 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.3 µs  57.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.4 µs  18.0%      1
_prepare_qkv                                                       2.7 µs  10.7%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.7 µs   7.0%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.6 µs   6.5%      1
total                                                             24.7 µs
GPU busy (kernels overlap: PDL or streams)                        23.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     347.8 µs  97.2%      1
_merge_norm_gate                                                   7.8 µs   2.2%      1
_prepare_qkv                                                       2.3 µs   0.6%      1
total                                                            358.0 µs
```

