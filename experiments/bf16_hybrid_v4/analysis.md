# bf16_hybrid_v4

Job: `ab09c58d874a41e0`
Correctness: True; official speedup: 1.10389x; CI: [1.087331214232024, 1.1204479009599029]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.21 | 12.69 | 1.119x | 7.87 |
| decode 1 | o | 6.66 | 6.04 | 1.103x | 3.50 |
| decode 1 | pooler | 21.34 | 17.71 | 1.205x | 13.99 |
| decode 1 | unpooler | 6.99 | 5.96 | 1.173x | 1.75 |
| decode 1 | router | 10.30 | 10.21 | 1.009x | 1.75 |
| decode 16 | qkv_gate | 13.84 | 13.92 | 0.994x | 7.93 |
| decode 16 | o | 6.85 | 6.82 | 1.005x | 3.56 |
| decode 16 | pooler | 21.34 | 20.21 | 1.056x | 14.06 |
| decode 16 | unpooler | 18.43 | 18.26 | 1.010x | 14.06 |
| decode 16 | router | 14.66 | 10.30 | 1.422x | 1.81 |
| decode 32 | qkv_gate | 13.87 | 13.95 | 0.994x | 8.00 |
| decode 32 | o | 6.82 | 6.92 | 0.985x | 3.62 |
| decode 32 | pooler | 22.08 | 22.22 | 0.994x | 14.14 |
| decode 32 | unpooler | 18.83 | 18.82 | 1.001x | 14.14 |
| decode 32 | router | 15.36 | 10.65 | 1.443x | 1.86 |
| verify 1 | qkv_gate | 13.78 | 13.84 | 0.996x | 7.90 |
| verify 1 | o | 6.46 | 6.46 | 1.000x | 3.53 |
| verify 1 | pooler | 21.41 | 19.99 | 1.071x | 14.02 |
| verify 1 | unpooler | 6.91 | 6.95 | 0.995x | 1.76 |
| verify 1 | router | 25.92 | 10.11 | 2.564x | 1.78 |
| verify 16 | qkv_gate | 17.22 | 17.07 | 1.008x | 8.42 |
| verify 16 | o | 7.41 | 7.26 | 1.021x | 3.99 |
| verify 16 | pooler | 24.88 | 25.14 | 0.990x | 14.64 |
| verify 16 | unpooler | 20.08 | 19.93 | 1.008x | 14.64 |
| verify 16 | router | 16.51 | 11.77 | 1.402x | 2.21 |
| verify 32 | qkv_gate | 20.10 | 19.87 | 1.011x | 9.77 |
| verify 32 | o | 9.25 | 9.18 | 1.008x | 4.48 |
| verify 32 | pooler | 30.82 | 31.06 | 0.992x | 17.37 |
| verify 32 | unpooler | 24.02 | 23.87 | 1.006x | 17.37 |
| verify 32 | router | 17.36 | 16.24 | 1.069x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.689 | 3.262 | 1.131x | 2.062x |
| decode 16 | 4.657 | 4.310 | 1.081x | 1.814x |
| decode 32 | 4.772 | 4.498 | 1.061x | 1.842x |
| verify 1 | 4.618 | 3.556 | 1.299x | 2.569x |
| verify 16 | 5.338 | 5.033 | 1.061x | 1.961x |
| verify 32 | 6.295 | 6.214 | 1.013x | 1.965x |

Recomputed geometric mean across loads: 1.103890x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              8.9 µs 100.0%      1
total                                                              8.9 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
_gemv                                                              4.4 µs 100.0%      1
total                                                              4.4 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                             17.8 µs 100.0%      1
total                                                             17.8 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                              3.2 µs 100.0%      1
total                                                              3.2 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  82.4%      1
_finish                                                            1.1 µs  17.6%      1
total                                                              6.2 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.1 µs  83.0%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  17.0%      1
total                                                              9.8 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                            4.8 µs 100.0%      1
total                                                              4.8 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.9 µs  93.1%      1
_finish                                                            1.4 µs   6.9%      1
total                                                             20.3 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.8 µs 100.0%      1
total                                                             18.8 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  80.8%      1
_finish                                                            1.2 µs  19.2%      1
total                                                              6.3 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     8.4 µs  81.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  18.1%      1
total                                                             10.3 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                            4.7 µs 100.0%      1
total                                                              4.7 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   19.7 µs  91.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs   8.8%      1
total                                                             21.6 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           18.5 µs 100.0%      1
total                                                             18.5 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.2 µs  80.8%      1
_finish                                                            1.2 µs  19.2%      1
total                                                              6.5 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     8.5 µs  83.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  16.1%      1
total                                                             10.1 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             4.4 µs 100.0%      1
total                                                              4.4 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.1 µs  93.2%      1
_finish                                                            1.3 µs   6.8%      1
total                                                             19.4 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.1 µs  72.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.5 µs  27.3%      1
total                                                              5.6 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  81.1%      1
_finish                                                            1.2 µs  18.9%      1
total                                                              6.3 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    13.5 µs  81.5%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs  18.5%      1
total                                                             16.6 µs
GPU busy (kernels overlap: PDL or streams)                        16.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            6.0 µs 100.0%      1
total                                                              6.0 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.0 µs 100.0%      1
total                                                             25.0 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           20.5 µs 100.0%      1
total                                                             20.5 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.4 µs  86.4%      1
_finish                                                            1.3 µs  13.6%      1
total                                                              9.7 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.3 µs 100.0%      1
total                                                             19.3 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.4 µs 100.0%      1
total                                                              8.4 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           30.1 µs 100.0%      1
total                                                             30.1 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          24.8 µs 100.0%      1
total                                                             24.8 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           14.2 µs  90.8%      1
_finish                                                            1.4 µs   9.2%      1
total                                                             15.7 µs
```

