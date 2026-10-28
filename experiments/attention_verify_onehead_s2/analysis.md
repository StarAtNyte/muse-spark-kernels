# attention_verify_onehead_s2

Job: `5971f63d8a544738`
Correctness: True; official speedup: 1.60467x; CI: [1.5806046510515772, 1.628744894230813]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.50 | 7.88 | 2.602x | 0.11 |
| decode 1 | full | 31.31 | 15.61 | 2.006x | 4.27 |
| decode 16 | sliding | 26.75 | 13.37 | 2.001x | 1.65 |
| decode 16 | full | 202.06 | 121.97 | 1.657x | 63.18 |
| decode 32 | sliding | 31.46 | 17.09 | 1.840x | 3.29 |
| decode 32 | full | 384.32 | 207.95 | 1.848x | 126.35 |
| verify 1 | sliding | 22.78 | 13.29 | 1.714x | 0.12 |
| verify 1 | full | 36.45 | 26.51 | 1.375x | 4.28 |
| verify 16 | sliding | 29.15 | 22.42 | 1.300x | 1.86 |
| verify 16 | full | 206.27 | 174.12 | 1.185x | 63.39 |
| verify 32 | sliding | 34.30 | 27.51 | 1.247x | 3.72 |
| verify 32 | full | 391.17 | 337.06 | 1.161x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.444 | 0.612 | 2.359x | 19.668x |
| decode 16 | 4.464 | 2.567 | 1.739x | 4.108x |
| decode 32 | 7.596 | 4.113 | 1.847x | 3.495x |
| verify 1 | 1.631 | 1.036 | 1.575x | 21.971x |
| verify 16 | 4.641 | 3.817 | 1.216x | 4.220x |
| verify 32 | 7.837 | 6.658 | 1.177x | 3.562x |

Recomputed geometric mean across loads: 1.604675x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  58.9%      1
_prepare_qkv                                                       1.6 µs  21.2%      1
_merge_norm_gate                                                   1.5 µs  19.9%      1
total                                                              7.6 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.5 µs  76.8%      1
_merge_norm_gate                                                   2.1 µs  15.5%      1
_prepare_qkv                                                       1.1 µs   7.7%      1
total                                                             13.7 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       7.1 µs  70.9%      1
_prepare_qkv                                                       1.5 µs  14.7%      1
_merge_norm_gate                                                   1.4 µs  14.4%      1
total                                                             10.0 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     115.8 µs  94.8%      1
_merge_norm_gate                                                   4.5 µs   3.7%      1
_prepare_qkv                                                       1.8 µs   1.5%      1
total                                                            122.1 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...      10.9 µs  76.1%      1
_prepare_qkv                                                       1.8 µs  12.5%      1
_merge_norm_gate                                                   1.6 µs  11.4%      1
total                                                             14.3 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     204.0 µs  97.4%      1
_merge_norm_gate                                                   3.5 µs   1.7%      1
_prepare_qkv                                                       2.0 µs   1.0%      1
total                                                            209.5 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.1 µs  66.4%      1
_merge_norm_gate                                                   1.7 µs  18.2%      1
_prepare_qkv                                                       1.4 µs  15.4%      1
total                                                              9.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 18.6 µs  82.5%      1
_merge_norm_gate                                                   2.7 µs  11.9%      1
_prepare_qkv                                                       1.2 µs   5.5%      1
total                                                             22.5 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  54.5%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.8 µs  19.2%      1
_prepare_qkv                                                       2.0 µs  10.0%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.9 µs   9.5%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.9%      1
total                                                             19.6 µs
GPU busy (kernels overlap: PDL or streams)                        18.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     167.8 µs  96.0%      1
_merge_norm_gate                                                   4.8 µs   2.8%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            174.8 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.7 µs  58.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs  16.4%      1
_prepare_qkv                                                       2.8 µs  11.0%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.0%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.7 µs   6.9%      1
total                                                             25.0 µs
GPU busy (kernels overlap: PDL or streams)                        23.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     322.0 µs  95.8%      1
_merge_norm_gate                                                  11.7 µs   3.5%      1
_prepare_qkv                                                       2.5 µs   0.7%      1
total                                                            336.1 µs
```

