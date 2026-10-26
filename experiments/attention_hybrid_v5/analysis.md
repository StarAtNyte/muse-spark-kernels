# attention_hybrid_v5

Job: `bbd722c637134459`
Correctness: True; official speedup: 1.23832x; CI: [1.219745660399295, 1.2568952744216086]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 21.07 | 10.68 | 1.973x | 0.11 |
| decode 1 | full | 31.49 | 22.39 | 1.406x | 4.27 |
| decode 16 | sliding | 26.93 | 20.93 | 1.287x | 1.65 |
| decode 16 | full | 200.96 | 185.87 | 1.081x | 63.18 |
| decode 32 | sliding | 30.98 | 24.79 | 1.250x | 3.29 |
| decode 32 | full | 383.33 | 356.71 | 1.075x | 126.35 |
| verify 1 | sliding | 23.15 | 14.39 | 1.609x | 0.12 |
| verify 1 | full | 36.29 | 30.12 | 1.205x | 4.28 |
| verify 16 | sliding | 29.15 | 22.61 | 1.289x | 1.86 |
| verify 16 | full | 205.60 | 198.99 | 1.033x | 63.39 |
| verify 32 | sliding | 34.22 | 27.76 | 1.233x | 3.72 |
| verify 32 | full | 390.46 | 383.25 | 1.019x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.473 | 0.850 | 1.734x | 20.067x |
| decode 16 | 4.454 | 3.937 | 1.131x | 4.099x |
| decode 32 | 7.558 | 6.848 | 1.104x | 3.478x |
| verify 1 | 1.646 | 1.144 | 1.439x | 22.165x |
| verify 16 | 4.631 | 4.224 | 1.096x | 4.210x |
| verify 32 | 7.822 | 7.409 | 1.056x | 3.556x |

Recomputed geometric mean across loads: 1.238320x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  4.4 µs  59.3%      1
_prepare_qkv                                                       1.5 µs  20.3%      1
_merge_norm_gate                                                   1.5 µs  20.3%      1
total                                                              7.4 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 14.4 µs  81.6%      1
_merge_norm_gate                                                   2.1 µs  12.2%      1
_prepare_qkv                                                       1.1 µs   6.2%      1
total                                                             17.6 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.1 µs  55.3%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.3 µs  18.2%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.8 µs  10.0%      1
_prepare_qkv                                                       1.8 µs   9.8%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   6.7%      1
total                                                             18.3 µs
GPU busy (kernels overlap: PDL or streams)                        16.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                181.3 µs  97.7%      1
_merge_norm_gate                                                   2.3 µs   1.2%      1
_prepare_qkv                                                       2.0 µs   1.1%      1
total                                                            185.6 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      13.2 µs  62.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.1 µs  14.7%      1
_prepare_qkv                                                       1.9 µs   9.0%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   8.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.0%      1
total                                                             21.3 µs
GPU busy (kernels overlap: PDL or streams)                        20.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                355.8 µs  98.8%      1
_merge_norm_gate                                                   2.3 µs   0.6%      1
_prepare_qkv                                                       2.0 µs   0.6%      1
total                                                            360.1 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.9 µs  69.0%      1
_merge_norm_gate                                                   1.6 µs  16.3%      1
_prepare_qkv                                                       1.5 µs  14.7%      1
total                                                             10.0 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      15.2 µs  54.3%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.6 µs  27.3%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       2.8 µs   9.8%      1
_prepare_qkv                                                       1.2 µs   4.5%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.2 µs   4.1%      1
total                                                             28.0 µs
GPU busy (kernels overlap: PDL or streams)                        26.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      10.7 µs  53.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.0 µs  20.1%      1
_prepare_qkv                                                       2.0 µs  10.1%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.9 µs   9.3%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   6.8%      1
total                                                             19.9 µs
GPU busy (kernels overlap: PDL or streams)                        18.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     193.4 µs  94.6%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs   2.0%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.1 µs   1.5%      1
_prepare_qkv                                                       2.1 µs   1.0%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.7 µs   0.8%      1
total                                                            204.4 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.2 µs  58.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.4 µs  17.8%      1
_prepare_qkv                                                       2.7 µs  11.0%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.8 µs   7.2%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   6.0%      1
total                                                             24.5 µs
GPU busy (kernels overlap: PDL or streams)                        23.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     377.0 µs  96.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.4 µs   1.1%      1
void flash::prepare_varlen_num_blocks_kernel<2, true>(int...       3.4 µs   0.9%      1
_prepare_qkv                                                       2.3 µs   0.6%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.9 µs   0.5%      1
total                                                            389.0 µs
```

