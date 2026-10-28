# projection_single_gemv_v2

Job: `be2868a5c1934c60`
Correctness: True; official speedup: 1.19482x; CI: [1.1768958723938812, 1.2127404167307505]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.87 | 10.26 | 1.353x | 7.87 |
| decode 1 | o | 6.85 | 6.11 | 1.120x | 3.50 |
| decode 1 | pooler | 21.41 | 17.71 | 1.209x | 13.99 |
| decode 1 | unpooler | 6.91 | 5.52 | 1.253x | 1.75 |
| decode 1 | router | 10.50 | 4.19 | 2.503x | 1.75 |
| decode 16 | qkv_gate | 13.92 | 14.17 | 0.982x | 7.93 |
| decode 16 | o | 6.62 | 6.70 | 0.989x | 3.56 |
| decode 16 | pooler | 21.79 | 19.58 | 1.113x | 14.06 |
| decode 16 | unpooler | 18.69 | 18.75 | 0.997x | 14.06 |
| decode 16 | router | 14.66 | 7.03 | 2.086x | 1.81 |
| decode 32 | qkv_gate | 14.29 | 14.69 | 0.972x | 8.00 |
| decode 32 | o | 6.62 | 6.66 | 0.995x | 3.62 |
| decode 32 | pooler | 21.78 | 21.87 | 0.996x | 14.14 |
| decode 32 | unpooler | 18.69 | 18.92 | 0.988x | 14.14 |
| decode 32 | router | 15.07 | 7.19 | 2.095x | 1.86 |
| verify 1 | qkv_gate | 14.00 | 14.14 | 0.990x | 7.90 |
| verify 1 | o | 6.37 | 6.42 | 0.993x | 3.53 |
| verify 1 | pooler | 21.36 | 16.72 | 1.277x | 14.02 |
| verify 1 | unpooler | 7.01 | 5.76 | 1.216x | 1.76 |
| verify 1 | router | 26.32 | 7.15 | 3.682x | 1.78 |
| verify 16 | qkv_gate | 16.91 | 16.90 | 1.001x | 8.42 |
| verify 16 | o | 7.23 | 7.30 | 0.990x | 3.99 |
| verify 16 | pooler | 25.07 | 25.01 | 1.003x | 14.64 |
| verify 16 | unpooler | 19.31 | 19.35 | 0.998x | 14.64 |
| verify 16 | router | 16.35 | 9.66 | 1.693x | 2.21 |
| verify 32 | qkv_gate | 19.62 | 19.71 | 0.995x | 9.77 |
| verify 32 | o | 9.25 | 9.31 | 0.993x | 4.48 |
| verify 32 | pooler | 30.43 | 30.46 | 0.999x | 17.37 |
| verify 32 | unpooler | 24.22 | 24.18 | 1.002x | 17.37 |
| verify 32 | router | 17.41 | 12.80 | 1.360x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.691 | 2.715 | 1.360x | 2.063x |
| decode 16 | 4.692 | 4.106 | 1.143x | 1.827x |
| decode 32 | 4.740 | 4.299 | 1.103x | 1.830x |
| verify 1 | 4.653 | 3.112 | 1.496x | 2.589x |
| verify 16 | 5.263 | 4.850 | 1.085x | 1.934x |
| verify 32 | 6.258 | 5.980 | 1.046x | 1.953x |

Recomputed geometric mean across loads: 1.194818x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              8.0 µs 100.0%      1
total                                                              8.0 µs
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
_gemv                                                             17.5 µs 100.0%      1
total                                                             17.5 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...       4.6 µs 100.0%      1
total                                                              4.6 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_gemv                                                              2.4 µs 100.0%      1
total                                                              2.4 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.2 µs  81.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  18.7%      1
total                                                             10.1 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                            4.6 µs 100.0%      1
total                                                              4.6 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<2, 16, 2, 4>(uint4 const*, __nv_bfloat16 ...      17.8 µs 100.0%      1
total                                                             17.8 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.3 µs 100.0%      1
total                                                             18.3 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.3 µs  67.1%      1
_finish                                                            1.6 µs  32.9%      1
total                                                              5.0 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     8.6 µs  81.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  18.2%      1
total                                                             10.5 µs
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
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.3 µs  91.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs   8.6%      1
total                                                             22.2 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           18.8 µs 100.0%      1
total                                                             18.8 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.3%      1
_finish                                                            1.7 µs  31.7%      1
total                                                              5.2 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     9.0 µs  84.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  15.6%      1
total                                                             10.7 µs
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
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      16.3 µs 100.0%      1
total                                                             16.3 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 32, 1, 3>(uint4 const*, __nv_bfloat16 ...       5.3 µs 100.0%      1
total                                                              5.3 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.5 µs  69.2%      1
_finish                                                            1.6 µs  30.8%      1
total                                                              5.1 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    12.9 µs  80.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs  19.2%      1
total                                                             16.0 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            5.3 µs 100.0%      1
total                                                              5.3 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           26.0 µs 100.0%      1
total                                                             26.0 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           21.3 µs 100.0%      1
total                                                             21.3 µs
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
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.2 µs 100.0%      1
total                                                             19.2 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.2 µs 100.0%      1
total                                                              8.2 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           27.6 µs 100.0%      1
total                                                             27.6 µs
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
_matmul                                                           10.2 µs  87.6%      1
_finish                                                            1.4 µs  12.4%      1
total                                                             11.6 µs
```

