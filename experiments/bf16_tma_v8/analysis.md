# bf16_tma_v8

Job: `b20bdf3f300043ef`
Correctness: True; official speedup: 1.10244x; CI: [1.0859054043755274, 1.1189786654224978]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.86 | 12.62 | 1.098x | 7.87 |
| decode 1 | o | 7.07 | 6.32 | 1.120x | 3.50 |
| decode 1 | pooler | 21.36 | 17.97 | 1.189x | 13.99 |
| decode 1 | unpooler | 6.88 | 5.88 | 1.170x | 1.75 |
| decode 1 | router | 10.34 | 7.00 | 1.477x | 1.75 |
| decode 16 | qkv_gate | 13.95 | 13.90 | 1.004x | 7.93 |
| decode 16 | o | 6.78 | 7.12 | 0.953x | 3.56 |
| decode 16 | pooler | 21.41 | 19.21 | 1.114x | 14.06 |
| decode 16 | unpooler | 18.50 | 18.52 | 0.999x | 14.06 |
| decode 16 | router | 15.14 | 7.02 | 2.157x | 1.81 |
| decode 32 | qkv_gate | 14.38 | 14.00 | 1.028x | 8.00 |
| decode 32 | o | 7.01 | 7.07 | 0.991x | 3.62 |
| decode 32 | pooler | 21.79 | 21.68 | 1.005x | 14.14 |
| decode 32 | unpooler | 18.78 | 18.78 | 1.000x | 14.14 |
| decode 32 | router | 15.26 | 7.09 | 2.152x | 1.86 |
| verify 1 | qkv_gate | 13.87 | 13.67 | 1.015x | 7.90 |
| verify 1 | o | 6.50 | 6.50 | 0.999x | 3.53 |
| verify 1 | pooler | 21.31 | 18.77 | 1.135x | 14.02 |
| verify 1 | unpooler | 7.20 | 7.14 | 1.009x | 1.76 |
| verify 1 | router | 25.98 | 6.77 | 3.841x | 1.78 |
| verify 16 | qkv_gate | 16.86 | 17.54 | 0.962x | 8.42 |
| verify 16 | o | 7.68 | 7.78 | 0.988x | 3.99 |
| verify 16 | pooler | 25.47 | 26.55 | 0.959x | 14.64 |
| verify 16 | unpooler | 20.02 | 26.87 | 0.745x | 14.64 |
| verify 16 | router | 16.32 | 9.65 | 1.691x | 2.21 |
| verify 32 | qkv_gate | 19.52 | 23.96 | 0.815x | 9.77 |
| verify 32 | o | 9.41 | 9.58 | 0.982x | 4.48 |
| verify 32 | pooler | 28.75 | 37.97 | 0.757x | 17.37 |
| verify 32 | unpooler | 24.00 | 34.14 | 0.703x | 17.37 |
| verify 32 | router | 17.28 | 12.60 | 1.371x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.689 | 3.086 | 1.195x | 2.062x |
| decode 16 | 4.698 | 4.078 | 1.152x | 1.829x |
| decode 32 | 4.788 | 4.255 | 1.125x | 1.849x |
| verify 1 | 4.642 | 3.276 | 1.417x | 2.583x |
| verify 16 | 5.354 | 5.480 | 0.977x | 1.967x |
| verify 32 | 6.136 | 7.331 | 0.837x | 1.915x |

Recomputed geometric mean across loads: 1.102442x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              8.5 µs 100.0%      1
total                                                              8.5 µs
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
_gemv                                                             17.1 µs 100.0%      1
total                                                             17.1 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                              3.6 µs 100.0%      1
total                                                              3.6 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.9%      1
_finish                                                            1.6 µs  31.1%      1
total                                                              5.2 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.0 µs  83.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  16.9%      1
total                                                              9.6 µs
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
_matmul                                                           18.0 µs  94.3%      1
_finish                                                            1.1 µs   5.7%      1
total                                                             19.1 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.5 µs 100.0%      1
total                                                             18.5 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.5 µs  68.5%      1
_finish                                                            1.6 µs  31.5%      1
total                                                              5.1 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     9.8 µs  84.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  15.9%      1
total                                                             11.6 µs
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
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.4 µs  91.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs   8.6%      1
total                                                             22.3 µs
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
_matmul                                                            3.7 µs  69.9%      1
_finish                                                            1.6 µs  30.1%      1
total                                                              5.3 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     8.5 µs  83.6%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  16.4%      1
total                                                             10.2 µs
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
_matmul                                                           17.6 µs  94.7%      1
_finish                                                            1.0 µs   5.3%      1
total                                                             18.6 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.0 µs  71.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  28.2%      1
total                                                              5.6 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.7%      1
_finish                                                            1.6 µs  31.3%      1
total                                                              5.2 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       16.0 µs  87.5%      1
_finish                                                            2.3 µs  12.5%      1
total                                                             18.2 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            5.4 µs 100.0%      1
total                                                              5.4 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       26.1 µs  91.1%      1
_finish                                                            2.6 µs   8.9%      1
total                                                             28.6 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       25.7 µs  90.7%      1
_finish                                                            2.6 µs   9.3%      1
total                                                             28.3 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.1 µs  77.4%      1
_finish                                                            1.8 µs  22.6%      1
total                                                              7.9 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       23.9 µs  90.7%      1
_finish                                                            2.5 µs   9.3%      1
total                                                             26.4 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.1 µs 100.0%      1
total                                                              8.1 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       38.6 µs  93.3%      1
_finish                                                            2.8 µs   6.7%      1
total                                                             41.4 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
_tma_matmul                                                       35.2 µs 100.0%      1
total                                                             35.2 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.1 µs  87.8%      1
_finish                                                            1.4 µs  12.2%      1
total                                                             11.5 µs
```

