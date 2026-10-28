# attention_cluster_sliding_v1

Job: `4448135fa8d44951`
Correctness: True; official speedup: 1.70623x; CI: [1.6806411985734546, 1.740507764222314]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.14 | 7.79 | 2.585x | 0.11 |
| decode 1 | full | 30.69 | 15.78 | 1.944x | 4.27 |
| decode 16 | sliding | 26.40 | 11.64 | 2.268x | 1.65 |
| decode 16 | full | 199.82 | 96.29 | 2.075x | 63.18 |
| decode 32 | sliding | 30.86 | 14.49 | 2.129x | 3.29 |
| decode 32 | full | 382.82 | 183.08 | 2.091x | 126.35 |
| verify 1 | sliding | 22.67 | 13.63 | 1.663x | 0.12 |
| verify 1 | full | 35.63 | 22.79 | 1.564x | 4.28 |
| verify 16 | sliding | 29.31 | 19.70 | 1.488x | 1.86 |
| verify 16 | full | 205.74 | 173.54 | 1.186x | 63.39 |
| verify 32 | sliding | 34.18 | 27.71 | 1.233x | 3.72 |
| verify 32 | full | 389.22 | 339.18 | 1.148x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.418 | 0.611 | 2.320x | 19.311x |
| decode 16 | 4.412 | 2.076 | 2.125x | 4.060x |
| decode 32 | 7.545 | 3.596 | 2.098x | 3.472x |
| verify 1 | 1.613 | 0.992 | 1.626x | 21.726x |
| verify 16 | 4.640 | 3.683 | 1.260x | 4.219x |
| verify 32 | 7.800 | 6.701 | 1.164x | 3.545x |

Recomputed geometric mean across loads: 1.706235x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.6%      1
_merge_norm_gate                                                   1.5 µs  21.1%      1
_prepare_qkv                                                       1.5 µs  20.3%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.2 µs  75.2%      1
_merge_norm_gate                                                   2.2 µs  16.0%      1
_prepare_qkv                                                       1.2 µs   8.7%      1
total                                                             13.6 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.3 µs  62.2%      1
_merge_norm_gate                                                   1.7 µs  19.9%      1
_prepare_qkv                                                       1.5 µs  18.0%      1
total                                                              8.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      93.3 µs  95.4%      1
_merge_norm_gate                                                   2.9 µs   3.0%      1
_prepare_qkv                                                       1.6 µs   1.6%      1
total                                                             97.8 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.7 µs  70.9%      1
_merge_norm_gate                                                   1.8 µs  14.7%      1
_prepare_qkv                                                       1.8 µs  14.4%      1
total                                                             12.2 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     180.1 µs  97.6%      1
_merge_norm_gate                                                   2.6 µs   1.4%      1
_prepare_qkv                                                       1.8 µs   1.0%      1
total                                                            184.5 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.3 µs  67.4%      1
_merge_norm_gate                                                   1.6 µs  17.5%      1
_prepare_qkv                                                       1.4 µs  15.1%      1
total                                                              9.3 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.2 µs  80.4%      1
_merge_norm_gate                                                   2.7 µs  13.5%      1
_prepare_qkv                                                       1.2 µs   6.1%      1
total                                                             20.1 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      14.7 µs  87.8%      1
_prepare_qkv                                                       2.0 µs  12.2%      1
total                                                             16.8 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     166.6 µs  95.9%      1
_merge_norm_gate                                                   5.1 µs   2.9%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            173.8 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.7 µs  59.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs  16.7%      1
_prepare_qkv                                                       2.5 µs  10.1%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.6 µs   6.6%      1
total                                                             24.8 µs
GPU busy (kernels overlap: PDL or streams)                        23.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     323.3 µs  95.9%      1
_merge_norm_gate                                                  11.6 µs   3.4%      1
_prepare_qkv                                                       2.3 µs   0.7%      1
total                                                            337.2 µs
```

