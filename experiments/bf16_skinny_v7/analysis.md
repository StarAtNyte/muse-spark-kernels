# bf16_skinny_v7

Job: `6dc8a03e41354b9b`
Correctness: True; official speedup: 1.13925x; CI: [1.1221615399705804, 1.1563390488021716]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.89 | 12.32 | 1.127x | 7.87 |
| decode 1 | o | 6.50 | 5.71 | 1.138x | 3.50 |
| decode 1 | pooler | 21.42 | 17.71 | 1.209x | 13.99 |
| decode 1 | unpooler | 6.98 | 5.83 | 1.196x | 1.75 |
| decode 1 | router | 10.45 | 6.88 | 1.518x | 1.75 |
| decode 16 | qkv_gate | 13.87 | 13.95 | 0.994x | 7.93 |
| decode 16 | o | 6.98 | 8.24 | 0.847x | 3.56 |
| decode 16 | pooler | 21.42 | 19.94 | 1.074x | 14.06 |
| decode 16 | unpooler | 18.69 | 20.16 | 0.927x | 14.06 |
| decode 16 | router | 15.04 | 7.33 | 2.052x | 1.81 |
| decode 32 | qkv_gate | 14.05 | 14.28 | 0.984x | 8.00 |
| decode 32 | o | 7.04 | 8.71 | 0.808x | 3.62 |
| decode 32 | pooler | 21.94 | 20.36 | 1.078x | 14.14 |
| decode 32 | unpooler | 18.82 | 20.57 | 0.915x | 14.14 |
| decode 32 | router | 15.17 | 7.20 | 2.107x | 1.86 |
| verify 1 | qkv_gate | 14.21 | 14.24 | 0.998x | 7.90 |
| verify 1 | o | 6.64 | 7.70 | 0.863x | 3.53 |
| verify 1 | pooler | 21.50 | 19.94 | 1.079x | 14.02 |
| verify 1 | unpooler | 7.01 | 6.97 | 1.006x | 1.76 |
| verify 1 | router | 26.14 | 6.94 | 3.766x | 1.78 |
| verify 16 | qkv_gate | 16.78 | 17.11 | 0.981x | 8.42 |
| verify 16 | o | 7.54 | 7.62 | 0.989x | 3.99 |
| verify 16 | pooler | 24.83 | 25.11 | 0.989x | 14.64 |
| verify 16 | unpooler | 20.02 | 20.38 | 0.982x | 14.64 |
| verify 16 | router | 16.32 | 9.63 | 1.694x | 2.21 |
| verify 32 | qkv_gate | 19.78 | 19.80 | 0.999x | 9.77 |
| verify 32 | o | 9.39 | 9.54 | 0.985x | 4.48 |
| verify 32 | pooler | 31.18 | 31.05 | 1.004x | 17.37 |
| verify 32 | unpooler | 24.51 | 24.83 | 0.987x | 17.37 |
| verify 32 | router | 17.89 | 13.20 | 1.356x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.672 | 3.005 | 1.222x | 2.053x |
| decode 16 | 4.712 | 4.316 | 1.092x | 1.835x |
| decode 32 | 4.774 | 4.409 | 1.083x | 1.843x |
| verify 1 | 4.681 | 3.458 | 1.354x | 2.605x |
| verify 16 | 5.300 | 4.951 | 1.071x | 1.948x |
| verify 32 | 6.371 | 6.101 | 1.044x | 1.989x |

Recomputed geometric mean across loads: 1.139250x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              9.3 µs 100.0%      1
total                                                              9.3 µs
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
_gemv                                                             17.3 µs 100.0%      1
total                                                             17.3 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                              3.1 µs 100.0%      1
total                                                              3.1 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  71.7%      1
_finish                                                            1.4 µs  28.3%      1
total                                                              5.1 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    10.6 µs  89.0%      1
_finish                                                            1.3 µs  11.0%      1
total                                                             11.9 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                     5.1 µs  81.9%      1
_finish                                                            1.1 µs  18.1%      1
total                                                              6.2 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    18.8 µs  94.4%      1
_finish                                                            1.1 µs   5.6%      1
total                                                             19.9 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    19.2 µs  94.3%      1
_finish                                                            1.2 µs   5.7%      1
total                                                             20.3 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.4 µs  69.5%      1
_finish                                                            1.5 µs  30.5%      1
total                                                              4.9 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    11.0 µs  88.7%      1
_finish                                                            1.4 µs  11.3%      1
total                                                             12.4 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                     5.2 µs  79.0%      1
_finish                                                            1.4 µs  21.0%      1
total                                                              6.6 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    19.5 µs  93.7%      1
_finish                                                            1.3 µs   6.3%      1
total                                                             20.8 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    19.2 µs  93.3%      1
_finish                                                            1.4 µs   6.7%      1
total                                                             20.6 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.7%      1
_finish                                                            1.7 µs  31.3%      1
total                                                              5.3 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    10.3 µs  88.7%      1
_finish                                                            1.3 µs  11.3%      1
total                                                             11.6 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                     5.1 µs  82.7%      1
_finish                                                            1.1 µs  17.3%      1
total                                                              6.1 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                    18.7 µs  94.8%      1
_finish                                                            1.0 µs   5.2%      1
total                                                             19.7 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
_skinny_matmul                                                     3.8 µs  69.6%      1
_finish                                                            1.7 µs  30.4%      1
total                                                              5.5 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  70.1%      1
_finish                                                            1.6 µs  29.9%      1
total                                                              5.2 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    12.8 µs  80.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.0 µs  19.2%      1
total                                                             15.8 µs
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
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.7 µs 100.0%      1
total                                                             25.7 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           21.0 µs 100.0%      1
total                                                             21.0 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.4 µs  78.4%      1
_finish                                                            1.8 µs  21.6%      1
total                                                              8.2 µs
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
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                           8.1 µs 100.0%      1
total                                                              8.1 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           30.4 µs 100.0%      1
total                                                             30.4 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          24.9 µs 100.0%      1
total                                                             24.9 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.3 µs  87.7%      1
_finish                                                            1.4 µs  12.3%      1
total                                                             11.7 µs
```

