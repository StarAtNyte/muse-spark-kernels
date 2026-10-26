# attention_fused_io_v1

Job: `9ba7233d35414c89`
Correctness: True; official speedup: 1.13356x; CI: [1.1165608668052354, 1.1505677967586943]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.59 | 16.11 | 1.278x | 0.11 |
| decode 1 | full | 31.42 | 27.18 | 1.156x | 4.27 |
| decode 16 | sliding | 27.18 | 21.00 | 1.294x | 1.65 |
| decode 16 | full | 201.30 | 195.47 | 1.030x | 63.18 |
| decode 32 | sliding | 31.07 | 25.25 | 1.231x | 3.29 |
| decode 32 | full | 385.68 | 379.93 | 1.015x | 126.35 |
| verify 1 | sliding | 22.53 | 16.57 | 1.360x | 0.12 |
| verify 1 | full | 36.42 | 30.49 | 1.194x | 4.28 |
| verify 16 | sliding | 29.18 | 22.70 | 1.285x | 1.86 |
| verify 16 | full | 206.16 | 199.26 | 1.035x | 63.39 |
| verify 32 | sliding | 34.46 | 28.03 | 1.230x | 3.72 |
| verify 32 | full | 390.61 | 382.88 | 1.020x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.450 | 1.176 | 1.233x | 19.752x |
| decode 16 | 4.471 | 4.093 | 1.092x | 4.115x |
| decode 32 | 7.600 | 7.241 | 1.050x | 3.497x |
| verify 1 | 1.619 | 1.250 | 1.295x | 21.806x |
| verify 16 | 4.641 | 4.233 | 1.097x | 4.219x |
| verify 32 | 7.835 | 7.415 | 1.057x | 3.562x |

Recomputed geometric mean across loads: 1.133564x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...       8.3 µs  51.6%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.2 µs  20.0%      1
_prepare_qkv                                                       1.8 µs  11.1%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  10.9%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.0 µs   6.4%      1
total                                                             16.1 µs
GPU busy (kernels overlap: PDL or streams)                        14.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.7 µs  62.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       6.9 µs  29.1%      1
_prepare_qkv                                                       1.1 µs   4.6%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.0 µs   4.3%      1
total                                                             23.7 µs
GPU busy (kernels overlap: PDL or streams)                        22.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...       9.8 µs  55.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.3 µs  18.4%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  10.3%      1
_prepare_qkv                                                       1.7 µs   9.4%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   6.7%      1
total                                                             17.7 µs
GPU busy (kernels overlap: PDL or streams)                        16.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     186.6 µs  96.3%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs   2.1%      1
_prepare_qkv                                                       1.7 µs   0.9%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.5 µs   0.8%      1
total                                                            193.9 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      13.3 µs  62.6%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.0 µs  14.2%      1
_prepare_qkv                                                       1.9 µs   8.9%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   8.3%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.0%      1
total                                                             21.2 µs
GPU busy (kernels overlap: PDL or streams)                        19.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     496.5 µs  98.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.5 µs   0.7%      1
_prepare_qkv                                                       1.9 µs   0.4%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.4 µs   0.3%      1
total                                                            503.3 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...       7.8 µs  48.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.7 µs  23.3%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  11.5%      1
_prepare_qkv                                                       1.5 µs   9.4%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.1 µs   7.0%      1
total                                                             15.9 µs
GPU busy (kernels overlap: PDL or streams)                        14.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.9 µs  54.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.6 µs  27.4%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       2.8 µs  10.1%      1
_prepare_qkv                                                       1.2 µs   4.4%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   4.2%      1
total                                                             27.7 µs
GPU busy (kernels overlap: PDL or streams)                        26.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.8 µs  54.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs  19.9%      1
_prepare_qkv                                                       1.9 µs   9.7%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.8%      1
total                                                             19.8 µs
GPU busy (kernels overlap: PDL or streams)                        18.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     187.8 µs  94.5%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.2 µs   2.1%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.0 µs   1.5%      1
_prepare_qkv                                                       2.0 µs   1.0%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.7 µs   0.8%      1
total                                                            198.7 µs
GPU busy (kernels overlap: PDL or streams)                       196.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.4 µs  58.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.4 µs  17.8%      1
_prepare_qkv                                                       2.6 µs  10.4%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.1%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   6.2%      1
total                                                             24.6 µs
GPU busy (kernels overlap: PDL or streams)                        23.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     374.1 µs  96.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.2 µs   1.1%      1
void flash::prepare_varlen_num_blocks_kernel<2, true>(int...       3.5 µs   0.9%      1
_prepare_qkv                                                       2.3 µs   0.6%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.9 µs   0.5%      1
total                                                            386.0 µs
```

