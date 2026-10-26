# bf16_hybrid_v5

Job: `8c1cc35a00be4e05`
Correctness: True; official speedup: 1.16890x; CI: [1.1513641167567037, 1.1864310441706134]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.13 | 12.44 | 1.136x | 7.87 |
| decode 1 | o | 6.50 | 5.54 | 1.173x | 3.50 |
| decode 1 | pooler | 21.34 | 17.53 | 1.217x | 13.99 |
| decode 1 | unpooler | 6.88 | 5.64 | 1.220x | 1.75 |
| decode 1 | router | 10.43 | 6.76 | 1.544x | 1.75 |
| decode 16 | qkv_gate | 14.08 | 13.88 | 1.015x | 7.93 |
| decode 16 | o | 6.75 | 6.51 | 1.037x | 3.56 |
| decode 16 | pooler | 21.28 | 18.77 | 1.134x | 14.06 |
| decode 16 | unpooler | 18.50 | 18.36 | 1.007x | 14.06 |
| decode 16 | router | 14.78 | 7.12 | 2.076x | 1.81 |
| decode 32 | qkv_gate | 14.21 | 14.15 | 1.004x | 8.00 |
| decode 32 | o | 6.82 | 6.91 | 0.986x | 3.62 |
| decode 32 | pooler | 21.92 | 22.04 | 0.995x | 14.14 |
| decode 32 | unpooler | 18.85 | 18.96 | 0.994x | 14.14 |
| decode 32 | router | 15.33 | 7.55 | 2.031x | 1.86 |
| verify 1 | qkv_gate | 14.05 | 13.95 | 1.007x | 7.90 |
| verify 1 | o | 6.62 | 6.74 | 0.983x | 3.53 |
| verify 1 | pooler | 21.28 | 19.08 | 1.115x | 14.02 |
| verify 1 | unpooler | 7.01 | 7.14 | 0.981x | 1.76 |
| verify 1 | router | 26.14 | 7.12 | 3.670x | 1.78 |
| verify 16 | qkv_gate | 17.10 | 16.92 | 1.011x | 8.42 |
| verify 16 | o | 7.26 | 7.37 | 0.986x | 3.99 |
| verify 16 | pooler | 25.33 | 24.78 | 1.022x | 14.64 |
| verify 16 | unpooler | 19.81 | 19.66 | 1.007x | 14.64 |
| verify 16 | router | 16.27 | 9.39 | 1.733x | 2.21 |
| verify 32 | qkv_gate | 19.89 | 19.90 | 0.999x | 9.77 |
| verify 32 | o | 9.18 | 9.17 | 1.001x | 4.48 |
| verify 32 | pooler | 30.70 | 30.74 | 0.999x | 17.37 |
| verify 32 | unpooler | 24.03 | 24.32 | 0.988x | 17.37 |
| verify 32 | router | 17.33 | 12.62 | 1.373x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.675 | 2.970 | 1.237x | 2.054x |
| decode 16 | 4.674 | 4.008 | 1.166x | 1.820x |
| decode 32 | 4.781 | 4.316 | 1.108x | 1.846x |
| verify 1 | 4.656 | 3.350 | 1.390x | 2.591x |
| verify 16 | 5.318 | 4.843 | 1.098x | 1.954x |
| verify 32 | 6.270 | 5.998 | 1.045x | 1.957x |

Recomputed geometric mean across loads: 1.168898x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              9.5 µs 100.0%      1
total                                                              9.5 µs
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
_matmul                                                            3.6 µs  70.8%      1
_finish                                                            1.5 µs  29.2%      1
total                                                              5.2 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.2 µs  82.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  17.1%      1
total                                                              9.9 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                            4.5 µs 100.0%      1
total                                                              4.5 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.9 µs  93.9%      1
_finish                                                            1.2 µs   6.1%      1
total                                                             19.0 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.4 µs 100.0%      1
total                                                             18.4 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.5%      1
_finish                                                            1.6 µs  31.5%      1
total                                                              5.2 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     8.3 µs  82.0%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.8 µs  18.0%      1
total                                                             10.1 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                            4.8 µs 100.0%      1
total                                                              4.8 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   19.9 µs  90.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       2.1 µs   9.6%      1
total                                                             22.0 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           19.3 µs 100.0%      1
total                                                             19.3 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  69.0%      1
_finish                                                            1.7 µs  31.0%      1
total                                                              5.4 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     9.5 µs  85.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  14.7%      1
total                                                             11.1 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             4.3 µs 100.0%      1
total                                                              4.3 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.4 µs  94.7%      1
_finish                                                            1.0 µs   5.3%      1
total                                                             19.5 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.0 µs  72.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.5 µs  27.9%      1
total                                                              5.5 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  69.9%      1
_finish                                                            1.6 µs  30.1%      1
total                                                              5.3 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    13.5 µs  81.6%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.0 µs  18.4%      1
total                                                             16.5 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            5.7 µs 100.0%      1
total                                                              5.7 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.4 µs 100.0%      1
total                                                             24.4 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           20.1 µs 100.0%      1
total                                                             20.1 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.5 µs  78.6%      1
_finish                                                            1.8 µs  21.4%      1
total                                                              8.2 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.9 µs 100.0%      1
total                                                             19.9 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.3 µs 100.0%      1
total                                                              8.3 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           28.8 µs 100.0%      1
total                                                             28.8 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          24.6 µs 100.0%      1
total                                                             24.6 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.3 µs  87.7%      1
_finish                                                            1.4 µs  12.3%      1
total                                                             11.7 µs
```

