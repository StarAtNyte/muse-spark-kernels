# projection_router_direct_v1

Job: `ee2cc07a65484f18`
Correctness: True; official speedup: 1.21337x; CI: [1.1951703519663013, 1.231571479437356]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.92 | 12.19 | 1.142x | 7.87 |
| decode 1 | o | 6.78 | 6.12 | 1.108x | 3.50 |
| decode 1 | pooler | 21.34 | 16.00 | 1.334x | 13.99 |
| decode 1 | unpooler | 7.07 | 5.79 | 1.222x | 1.75 |
| decode 1 | router | 10.27 | 3.98 | 2.582x | 1.75 |
| decode 16 | qkv_gate | 14.59 | 13.93 | 1.048x | 7.93 |
| decode 16 | o | 6.56 | 6.50 | 1.010x | 3.56 |
| decode 16 | pooler | 21.15 | 18.60 | 1.137x | 14.06 |
| decode 16 | unpooler | 18.35 | 18.20 | 1.008x | 14.06 |
| decode 16 | router | 15.01 | 6.84 | 2.194x | 1.81 |
| decode 32 | qkv_gate | 15.33 | 14.59 | 1.051x | 8.00 |
| decode 32 | o | 6.85 | 6.79 | 1.008x | 3.62 |
| decode 32 | pooler | 21.70 | 21.57 | 1.006x | 14.14 |
| decode 32 | unpooler | 18.27 | 18.07 | 1.011x | 14.14 |
| decode 32 | router | 15.17 | 7.03 | 2.158x | 1.86 |
| verify 1 | qkv_gate | 14.08 | 13.26 | 1.062x | 7.90 |
| verify 1 | o | 6.85 | 6.79 | 1.008x | 3.53 |
| verify 1 | pooler | 21.14 | 16.62 | 1.272x | 14.02 |
| verify 1 | unpooler | 7.17 | 5.94 | 1.207x | 1.76 |
| verify 1 | router | 26.56 | 7.07 | 3.759x | 1.78 |
| verify 16 | qkv_gate | 18.14 | 17.93 | 1.012x | 8.42 |
| verify 16 | o | 7.33 | 7.43 | 0.986x | 3.99 |
| verify 16 | pooler | 26.10 | 25.30 | 1.031x | 14.64 |
| verify 16 | unpooler | 20.11 | 19.95 | 1.008x | 14.64 |
| verify 16 | router | 16.66 | 9.81 | 1.698x | 2.21 |
| verify 32 | qkv_gate | 19.81 | 19.39 | 1.022x | 9.77 |
| verify 32 | o | 9.28 | 9.42 | 0.985x | 4.48 |
| verify 32 | pooler | 30.21 | 30.09 | 1.004x | 17.37 |
| verify 32 | unpooler | 24.26 | 24.22 | 1.002x | 17.37 |
| verify 32 | router | 17.68 | 13.00 | 1.360x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.682 | 2.733 | 1.348x | 2.058x |
| decode 16 | 4.691 | 3.972 | 1.181x | 1.827x |
| decode 32 | 4.793 | 4.219 | 1.136x | 1.851x |
| verify 1 | 4.699 | 3.080 | 1.526x | 2.615x |
| verify 16 | 5.477 | 4.986 | 1.098x | 2.012x |
| verify 32 | 6.276 | 5.959 | 1.053x | 1.959x |

Recomputed geometric mean across loads: 1.213371x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              9.4 µs 100.0%      1
total                                                              9.4 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
_gemv                                                              4.3 µs 100.0%      1
total                                                              4.3 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      15.2 µs 100.0%      1
total                                                             15.2 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 32, 1, 4>(uint4 const*, __nv_bfloat16 ...       4.9 µs 100.0%      1
total                                                              4.9 µs
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
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     8.0 µs  81.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.8 µs  18.6%      1
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
_matmul                                                           18.3 µs  94.4%      1
_finish                                                            1.1 µs   5.6%      1
total                                                             19.4 µs
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
_matmul                                                            3.6 µs  67.7%      1
_finish                                                            1.7 µs  32.3%      1
total                                                              5.2 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     9.3 µs  83.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.9 µs  16.7%      1
total                                                             11.1 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                            5.0 µs 100.0%      1
total                                                              5.0 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.7 µs  91.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       2.0 µs   8.8%      1
total                                                             22.7 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           18.6 µs 100.0%      1
total                                                             18.6 µs
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
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      10.9 µs 100.0%      1
total                                                             10.9 µs
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
void bf16c_gemm<1, 16, 1, 4>(uint4 const*, __nv_bfloat16 ...      16.4 µs 100.0%      1
total                                                             16.4 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
void bf16c_gemm<1, 32, 1, 3>(uint4 const*, __nv_bfloat16 ...       5.1 µs 100.0%      1
total                                                              5.1 µs
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
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    13.0 µs  80.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs  19.1%      1
total                                                             16.1 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                            5.5 µs 100.0%      1
total                                                              5.5 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           26.6 µs 100.0%      1
total                                                             26.6 µs
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
_matmul                                                            6.2 µs  78.0%      1
_finish                                                            1.8 µs  22.0%      1
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
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           31.6 µs 100.0%      1
total                                                             31.6 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          24.4 µs 100.0%      1
total                                                             24.4 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.8 µs  88.3%      1
_finish                                                            1.4 µs  11.7%      1
total                                                             12.3 µs
```

