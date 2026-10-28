# bf16c_v1

Job: `5599a49bc7204354`
Correctness: True; official speedup: 1.18288x; CI: [1.1651349245430633, 1.2006212674225474]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.08 | 12.49 | 1.128x | 7.87 |
| decode 1 | o | 6.88 | 6.04 | 1.139x | 3.50 |
| decode 1 | pooler | 21.09 | 16.03 | 1.315x | 13.99 |
| decode 1 | unpooler | 7.23 | 5.91 | 1.224x | 1.75 |
| decode 1 | router | 10.53 | 6.96 | 1.513x | 1.75 |
| decode 16 | qkv_gate | 14.10 | 14.14 | 0.997x | 7.93 |
| decode 16 | o | 6.90 | 6.86 | 1.006x | 3.56 |
| decode 16 | pooler | 21.38 | 18.90 | 1.131x | 14.06 |
| decode 16 | unpooler | 18.66 | 18.58 | 1.004x | 14.06 |
| decode 16 | router | 14.94 | 7.15 | 2.089x | 1.81 |
| decode 32 | qkv_gate | 14.58 | 14.60 | 0.998x | 8.00 |
| decode 32 | o | 6.72 | 6.69 | 1.005x | 3.62 |
| decode 32 | pooler | 21.81 | 21.55 | 1.012x | 14.14 |
| decode 32 | unpooler | 18.74 | 18.62 | 1.006x | 14.14 |
| decode 32 | router | 15.17 | 7.11 | 2.134x | 1.86 |
| verify 1 | qkv_gate | 14.02 | 14.24 | 0.985x | 7.90 |
| verify 1 | o | 6.62 | 6.51 | 1.017x | 3.53 |
| verify 1 | pooler | 21.28 | 16.47 | 1.292x | 14.02 |
| verify 1 | unpooler | 6.94 | 6.98 | 0.994x | 1.76 |
| verify 1 | router | 26.43 | 7.25 | 3.646x | 1.78 |
| verify 16 | qkv_gate | 17.07 | 17.02 | 1.003x | 8.42 |
| verify 16 | o | 7.30 | 7.34 | 0.994x | 3.99 |
| verify 16 | pooler | 25.14 | 25.08 | 1.002x | 14.64 |
| verify 16 | unpooler | 19.57 | 19.52 | 1.003x | 14.64 |
| verify 16 | router | 16.18 | 9.51 | 1.701x | 2.21 |
| verify 32 | qkv_gate | 19.87 | 19.69 | 1.009x | 9.77 |
| verify 32 | o | 9.38 | 9.31 | 1.007x | 4.48 |
| verify 32 | pooler | 30.74 | 30.69 | 1.002x | 17.37 |
| verify 32 | unpooler | 24.45 | 24.37 | 1.003x | 17.37 |
| verify 32 | router | 17.44 | 12.84 | 1.358x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.708 | 2.941 | 1.261x | 2.073x |
| decode 16 | 4.710 | 4.070 | 1.157x | 1.834x |
| decode 32 | 4.774 | 4.251 | 1.123x | 1.843x |
| verify 1 | 4.668 | 3.190 | 1.463x | 2.597x |
| verify 16 | 5.285 | 4.865 | 1.086x | 1.942x |
| verify 32 | 6.316 | 6.008 | 1.051x | 1.972x |

Recomputed geometric mean across loads: 1.182878x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              9.2 µs 100.0%      1
total                                                              9.2 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
_gemv                                                              4.5 µs 100.0%      1
total                                                              4.5 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      14.9 µs 100.0%      1
total                                                             14.9 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 32, 1, 4>(uint4 const*, __nv_bfloat16 ...       5.1 µs 100.0%      1
total                                                              5.1 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.5 µs  70.5%      1
_finish                                                            1.5 µs  29.5%      1
total                                                              5.0 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.2 µs  83.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  16.8%      1
total                                                              9.9 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                            4.2 µs 100.0%      1
total                                                              4.2 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.5 µs  94.2%      1
_finish                                                            1.1 µs   5.8%      1
total                                                             18.6 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.1 µs 100.0%      1
total                                                             18.1 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.5 µs  68.3%      1
_finish                                                            1.6 µs  31.7%      1
total                                                              5.2 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    10.6 µs  83.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       2.0 µs  16.2%      1
total                                                             12.6 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                            4.9 µs 100.0%      1
total                                                              4.9 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   19.8 µs  91.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs   8.7%      1
total                                                             21.7 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           19.2 µs 100.0%      1
total                                                             19.2 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  68.9%      1
_finish                                                            1.7 µs  31.1%      1
total                                                              5.3 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     9.2 µs  84.6%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  15.4%      1
total                                                             10.8 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             4.2 µs 100.0%      1
total                                                              4.2 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      16.6 µs 100.0%      1
total                                                             16.6 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.1 µs  72.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  27.8%      1
total                                                              5.6 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.9%      1
_finish                                                            1.6 µs  31.1%      1
total                                                              5.2 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    12.9 µs  81.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.0 µs  18.8%      1
total                                                             15.8 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            6.1 µs 100.0%      1
total                                                              6.1 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.2 µs 100.0%      1
total                                                             25.2 µs
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
_matmul                                                            6.2 µs  77.9%      1
_finish                                                            1.8 µs  22.1%      1
total                                                              8.0 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.6 µs 100.0%      1
total                                                             19.6 µs
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
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           30.3 µs 100.0%      1
total                                                             30.3 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          24.5 µs 100.0%      1
total                                                             24.5 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.2 µs  87.8%      1
_finish                                                            1.4 µs  12.2%      1
total                                                             11.6 µs
```

