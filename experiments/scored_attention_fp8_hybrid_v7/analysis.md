# scored_attention_fp8_hybrid_v7

Job: `b9b6263e128341ad`
Correctness: True; official speedup: 1.32482x; CI: [1.3049473043715603, 1.344691892321963]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.42 | 10.02 | 2.038x | 0.11 |
| decode 1 | full | 31.63 | 21.09 | 1.500x | 4.27 |
| decode 16 | sliding | 26.42 | 20.26 | 1.304x | 1.65 |
| decode 16 | full | 200.86 | 156.26 | 1.285x | 63.18 |
| decode 32 | sliding | 31.54 | 23.45 | 1.345x | 3.29 |
| decode 32 | full | 382.82 | 301.51 | 1.270x | 126.35 |
| verify 1 | sliding | 22.99 | 13.69 | 1.680x | 0.12 |
| verify 1 | full | 36.38 | 26.48 | 1.374x | 4.28 |
| verify 16 | sliding | 29.42 | 22.85 | 1.288x | 1.86 |
| verify 16 | full | 204.91 | 197.95 | 1.035x | 63.39 |
| verify 32 | sliding | 34.27 | 27.89 | 1.229x | 3.72 |
| verify 32 | full | 389.31 | 382.46 | 1.018x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.445 | 0.798 | 1.811x | 19.687x |
| decode 16 | 4.429 | 3.432 | 1.291x | 4.076x |
| decode 32 | 7.576 | 5.903 | 1.283x | 3.486x |
| verify 1 | 1.640 | 1.053 | 1.557x | 22.086x |
| verify 16 | 4.632 | 4.218 | 1.098x | 4.211x |
| verify 32 | 7.806 | 7.402 | 1.054x | 3.548x |

Recomputed geometric mean across loads: 1.324820x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  4.1 µs  54.2%      1
_prepare_qkv                                                       2.0 µs  25.8%      1
_merge_norm_gate                                                   1.5 µs  19.9%      1
total                                                              7.6 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 12.3 µs  79.4%      1
_merge_norm_gate                                                   2.1 µs  13.6%      1
_prepare_qkv                                                       1.1 µs   7.0%      1
total                                                             15.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...       9.7 µs  54.5%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.3 µs  18.3%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  10.2%      1
_prepare_qkv                                                       1.7 µs   9.7%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   7.3%      1
total                                                             17.9 µs
GPU busy (kernels overlap: PDL or streams)                        16.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                154.0 µs  97.3%      1
_merge_norm_gate                                                   2.3 µs   1.5%      1
_prepare_qkv                                                       1.9 µs   1.2%      1
total                                                            158.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 14.5 µs  81.6%      1
_prepare_qkv                                                       1.6 µs   9.2%      1
_merge_norm_gate                                                   1.6 µs   9.2%      1
total                                                             17.7 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                298.9 µs  98.6%      1
_merge_norm_gate                                                   2.2 µs   0.7%      1
_prepare_qkv                                                       2.1 µs   0.7%      1
total                                                            303.2 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.0 µs  66.0%      1
_merge_norm_gate                                                   1.7 µs  18.2%      1
_prepare_qkv                                                       1.4 µs  15.8%      1
total                                                              9.1 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 18.8 µs  82.9%      1
_merge_norm_gate                                                   2.7 µs  11.9%      1
_prepare_qkv                                                       1.2 µs   5.2%      1
total                                                             22.7 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.8 µs  54.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.8 µs  19.4%      1
_prepare_qkv                                                       2.0 µs   9.9%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs   9.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.8%      1
total                                                             19.8 µs
GPU busy (kernels overlap: PDL or streams)                        18.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     190.9 µs  94.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.8 µs   1.9%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.1 µs   1.6%      1
_prepare_qkv                                                       1.8 µs   0.9%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.7 µs   0.8%      1
total                                                            201.3 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.2 µs  57.6%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.5 µs  18.2%      1
_prepare_qkv                                                       2.5 µs  10.2%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.3%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.7 µs   6.7%      1
total                                                             24.7 µs
GPU busy (kernels overlap: PDL or streams)                        23.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     374.9 µs  97.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs   1.0%      1
void flash::prepare_varlen_num_blocks_kernel<2, true>(int...       3.5 µs   0.9%      1
_prepare_qkv                                                       2.3 µs   0.6%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       2.0 µs   0.5%      1
total                                                            386.6 µs
```

