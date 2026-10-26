# scored_bf16_v5

Job: `3ee947021eae43b9`
Correctness: True; official speedup: 1.17430x; CI: [1.156684964359401, 1.1919139480454741]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.16 | 11.99 | 1.180x | 7.87 |
| decode 1 | o | 6.62 | 5.78 | 1.146x | 3.50 |
| decode 1 | pooler | 21.31 | 17.69 | 1.205x | 13.99 |
| decode 1 | unpooler | 7.09 | 6.02 | 1.177x | 1.75 |
| decode 1 | router | 10.30 | 6.81 | 1.514x | 1.75 |
| decode 16 | qkv_gate | 14.18 | 13.46 | 1.053x | 7.93 |
| decode 16 | o | 7.14 | 7.23 | 0.987x | 3.56 |
| decode 16 | pooler | 21.57 | 19.18 | 1.125x | 14.06 |
| decode 16 | unpooler | 18.72 | 18.62 | 1.006x | 14.06 |
| decode 16 | router | 14.96 | 7.26 | 2.062x | 1.81 |
| decode 32 | qkv_gate | 14.80 | 14.15 | 1.046x | 8.00 |
| decode 32 | o | 7.12 | 6.95 | 1.025x | 3.62 |
| decode 32 | pooler | 21.78 | 21.50 | 1.013x | 14.14 |
| decode 32 | unpooler | 18.90 | 18.76 | 1.007x | 14.14 |
| decode 32 | router | 15.33 | 7.25 | 2.113x | 1.86 |
| verify 1 | qkv_gate | 14.38 | 14.06 | 1.023x | 7.90 |
| verify 1 | o | 6.75 | 6.51 | 1.037x | 3.53 |
| verify 1 | pooler | 21.26 | 18.71 | 1.136x | 14.02 |
| verify 1 | unpooler | 7.17 | 7.04 | 1.019x | 1.76 |
| verify 1 | router | 26.14 | 6.87 | 3.806x | 1.78 |
| verify 16 | qkv_gate | 17.22 | 16.57 | 1.039x | 8.42 |
| verify 16 | o | 7.49 | 7.47 | 1.002x | 3.99 |
| verify 16 | pooler | 25.57 | 25.60 | 0.999x | 14.64 |
| verify 16 | unpooler | 19.90 | 19.89 | 1.001x | 14.64 |
| verify 16 | router | 16.35 | 9.70 | 1.686x | 2.21 |
| verify 32 | qkv_gate | 19.58 | 19.79 | 0.989x | 9.77 |
| verify 32 | o | 9.28 | 9.41 | 0.986x | 4.48 |
| verify 32 | pooler | 30.82 | 31.17 | 0.989x | 17.37 |
| verify 32 | unpooler | 23.98 | 24.50 | 0.979x | 17.37 |
| verify 32 | router | 17.60 | 12.92 | 1.362x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.688 | 2.995 | 1.232x | 2.062x |
| decode 16 | 4.747 | 4.076 | 1.165x | 1.848x |
| decode 32 | 4.831 | 4.254 | 1.136x | 1.865x |
| verify 1 | 4.694 | 3.298 | 1.423x | 2.612x |
| verify 16 | 5.365 | 4.912 | 1.092x | 1.971x |
| verify 32 | 6.278 | 6.063 | 1.035x | 1.960x |

Recomputed geometric mean across loads: 1.174299x.
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
_gemv                                                              4.6 µs 100.0%      1
total                                                              4.6 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                             17.6 µs 100.0%      1
total                                                             17.6 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                              3.4 µs 100.0%      1
total                                                              3.4 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  70.0%      1
_finish                                                            1.5 µs  30.0%      1
total                                                              5.1 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.2 µs  83.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  16.7%      1
total                                                              9.8 µs
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
_matmul                                                           17.8 µs  93.8%      1
_finish                                                            1.2 µs   6.2%      1
total                                                             19.0 µs
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
_matmul                                                            3.5 µs  68.1%      1
_finish                                                            1.6 µs  31.9%      1
total                                                              5.1 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     8.6 µs  82.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  17.7%      1
total                                                             10.5 µs
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
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   19.9 µs  91.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       2.0 µs   8.9%      1
total                                                             21.8 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           19.6 µs 100.0%      1
total                                                             19.6 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.9%      1
_finish                                                            1.6 µs  31.1%      1
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
_matmul                                                           17.8 µs  94.4%      1
_finish                                                            1.1 µs   5.6%      1
total                                                             18.8 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.1 µs  73.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.5 µs  26.6%      1
total                                                              5.5 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  69.6%      1
_finish                                                            1.6 µs  30.4%      1
total                                                              5.2 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    13.2 µs  81.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.0 µs  18.6%      1
total                                                             16.2 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            5.6 µs 100.0%      1
total                                                              5.6 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           23.9 µs 100.0%      1
total                                                             23.9 µs
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
_matmul                                                            6.3 µs  78.6%      1
_finish                                                            1.7 µs  21.4%      1
total                                                              8.1 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.5 µs 100.0%      1
total                                                             19.5 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.5 µs 100.0%      1
total                                                              8.5 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           29.3 µs 100.0%      1
total                                                             29.3 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          25.6 µs 100.0%      1
total                                                             25.6 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.0 µs  87.7%      1
_finish                                                            1.4 µs  12.3%      1
total                                                             11.4 µs
```

