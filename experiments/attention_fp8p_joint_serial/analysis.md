# attention_fp8p_joint_serial

Job: `cf508e81d0ae4849`
Correctness: True; official speedup: 1.71123x; CI: [1.6855585457826436, 1.7442675370871945]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.62 | 7.99 | 2.580x | 0.11 |
| decode 1 | full | 31.54 | 16.01 | 1.970x | 4.27 |
| decode 16 | sliding | 26.80 | 11.97 | 2.240x | 1.65 |
| decode 16 | full | 199.87 | 96.43 | 2.073x | 63.18 |
| decode 32 | sliding | 31.20 | 14.79 | 2.109x | 3.29 |
| decode 32 | full | 384.06 | 183.80 | 2.090x | 126.35 |
| verify 1 | sliding | 23.14 | 13.78 | 1.679x | 0.12 |
| verify 1 | full | 36.45 | 22.88 | 1.593x | 4.28 |
| verify 16 | sliding | 29.30 | 22.64 | 1.294x | 1.86 |
| verify 16 | full | 204.43 | 166.36 | 1.229x | 63.39 |
| verify 32 | sliding | 34.62 | 27.76 | 1.247x | 3.72 |
| verify 32 | full | 389.73 | 333.51 | 1.169x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.453 | 0.624 | 2.330x | 19.797x |
| decode 16 | 4.431 | 2.093 | 2.117x | 4.078x |
| decode 32 | 7.580 | 3.621 | 2.093x | 3.488x |
| verify 1 | 1.647 | 1.000 | 1.648x | 22.189x |
| verify 16 | 4.619 | 3.703 | 1.247x | 4.199x |
| verify 32 | 7.828 | 6.613 | 1.184x | 3.559x |

Recomputed geometric mean across loads: 1.711227x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  59.3%      1
_prepare_qkv                                                       1.5 µs  20.3%      1
_merge_norm_gate                                                   1.5 µs  20.3%      1
total                                                              7.4 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.4 µs  76.6%      1
_merge_norm_gate                                                   2.1 µs  15.6%      1
_prepare_qkv                                                       1.1 µs   7.8%      1
total                                                             13.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.4 µs  61.8%      1
_merge_norm_gate                                                   1.7 µs  19.9%      1
_prepare_qkv                                                       1.6 µs  18.4%      1
total                                                              8.7 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      90.8 µs  95.1%      1
_merge_norm_gate                                                   3.0 µs   3.1%      1
_prepare_qkv                                                       1.7 µs   1.8%      1
total                                                             95.6 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.4 µs  69.8%      1
_merge_norm_gate                                                   1.9 µs  15.4%      1
_prepare_qkv                                                       1.8 µs  14.8%      1
total                                                             12.1 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     181.1 µs  97.6%      1
_merge_norm_gate                                                   2.6 µs   1.4%      1
_prepare_qkv                                                       1.9 µs   1.0%      1
total                                                            185.5 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.1 µs  66.2%      1
_merge_norm_gate                                                   1.7 µs  17.9%      1
_prepare_qkv                                                       1.5 µs  15.9%      1
total                                                              9.3 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.4 µs  80.9%      1
_merge_norm_gate                                                   2.7 µs  13.1%      1
_prepare_qkv                                                       1.2 µs   6.0%      1
total                                                             20.2 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  53.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs  19.6%      1
_prepare_qkv                                                       2.0 µs  10.4%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.4 µs   7.0%      1
total                                                             19.8 µs
GPU busy (kernels overlap: PDL or streams)                        18.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     159.9 µs  95.7%      1
_merge_norm_gate                                                   4.9 µs   2.9%      1
_prepare_qkv                                                       2.3 µs   1.4%      1
total                                                            167.1 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.5 µs  58.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.5 µs  18.0%      1
_prepare_qkv                                                       2.6 µs  10.4%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   5.9%      1
total                                                             24.9 µs
GPU busy (kernels overlap: PDL or streams)                        23.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     322.3 µs  95.9%      1
_merge_norm_gate                                                  11.6 µs   3.4%      1
_prepare_qkv                                                       2.3 µs   0.7%      1
total                                                            336.2 µs
```

