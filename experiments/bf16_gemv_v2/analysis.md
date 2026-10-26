# bf16_gemv_v2

Job: `22bdb0294f8c4816`
Correctness: True; official speedup: 1.03168x; CI: [1.0162082962928283, 1.0471588027789043]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.24 | 12.63 | 1.127x | 7.87 |
| decode 1 | o | 6.69 | 6.03 | 1.110x | 3.50 |
| decode 1 | pooler | 21.30 | 17.73 | 1.201x | 13.99 |
| decode 1 | unpooler | 7.04 | 5.91 | 1.192x | 1.75 |
| decode 1 | router | 10.58 | 18.37 | 0.576x | 1.75 |
| decode 16 | qkv_gate | 13.94 | 14.38 | 0.969x | 7.93 |
| decode 16 | o | 6.91 | 8.76 | 0.789x | 3.56 |
| decode 16 | pooler | 21.49 | 20.15 | 1.066x | 14.06 |
| decode 16 | unpooler | 18.34 | 20.69 | 0.886x | 14.06 |
| decode 16 | router | 14.94 | 10.44 | 1.431x | 1.81 |
| decode 32 | qkv_gate | 14.21 | 15.29 | 0.929x | 8.00 |
| decode 32 | o | 6.75 | 11.66 | 0.579x | 3.62 |
| decode 32 | pooler | 21.95 | 22.08 | 0.994x | 14.14 |
| decode 32 | unpooler | 18.74 | 25.61 | 0.731x | 14.14 |
| decode 32 | router | 15.22 | 10.31 | 1.476x | 1.86 |
| verify 1 | qkv_gate | 14.37 | 14.60 | 0.984x | 7.90 |
| verify 1 | o | 6.62 | 8.15 | 0.812x | 3.53 |
| verify 1 | pooler | 21.50 | 20.25 | 1.062x | 14.02 |
| verify 1 | unpooler | 7.04 | 6.98 | 1.009x | 1.76 |
| verify 1 | router | 26.59 | 10.67 | 2.491x | 1.78 |
| verify 16 | qkv_gate | 16.94 | 16.68 | 1.016x | 8.42 |
| verify 16 | o | 7.36 | 7.55 | 0.975x | 3.99 |
| verify 16 | pooler | 25.55 | 25.96 | 0.984x | 14.64 |
| verify 16 | unpooler | 19.87 | 19.98 | 0.994x | 14.64 |
| verify 16 | router | 16.43 | 11.77 | 1.396x | 2.21 |
| verify 32 | qkv_gate | 19.84 | 19.61 | 1.012x | 9.77 |
| verify 32 | o | 9.52 | 9.69 | 0.982x | 4.48 |
| verify 32 | pooler | 30.69 | 30.81 | 0.996x | 17.37 |
| verify 32 | unpooler | 24.32 | 24.58 | 0.989x | 17.37 |
| verify 32 | router | 17.78 | 16.66 | 1.067x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.710 | 3.761 | 0.986x | 2.074x |
| decode 16 | 4.688 | 4.615 | 1.016x | 1.825x |
| decode 32 | 4.766 | 5.267 | 0.905x | 1.840x |
| verify 1 | 4.720 | 3.761 | 1.255x | 2.626x |
| verify 16 | 5.342 | 5.081 | 1.051x | 1.963x |
| verify 32 | 6.333 | 6.284 | 1.008x | 1.977x |

Recomputed geometric mean across loads: 1.031684x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_gemv                                                              8.8 µs 100.0%      1
total                                                              8.8 µs
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
_gemv                                                             17.4 µs 100.0%      1
total                                                             17.4 µs
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
_gemv                                                              8.3 µs 100.0%      1
total                                                              8.3 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.9 µs  87.1%      1
_finish                                                            1.3 µs  12.9%      1
total                                                             10.2 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.3 µs  76.0%      1
_finish                                                            1.7 µs  24.0%      1
total                                                              6.9 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.6 µs  93.0%      1
_finish                                                            1.4 µs   7.0%      1
total                                                             20.0 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           20.3 µs  92.3%      1
_finish                                                            1.7 µs   7.7%      1
total                                                             22.0 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  80.2%      1
_finish                                                            1.2 µs  19.8%      1
total                                                              6.3 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           12.6 µs  89.3%      1
_finish                                                            1.5 µs  10.7%      1
total                                                             14.1 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.4 µs  79.5%      1
_finish                                                            2.2 µs  20.5%      1
total                                                             10.6 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           20.6 µs  92.1%      1
_finish                                                            1.8 µs   7.9%      1
total                                                             22.4 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           23.7 µs  90.0%      1
_finish                                                            2.6 µs  10.0%      1
total                                                             26.3 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  80.4%      1
_finish                                                            1.2 µs  19.6%      1
total                                                              6.4 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.3 µs  86.6%      1
_finish                                                            1.3 µs  13.4%      1
total                                                              9.6 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  77.8%      1
_finish                                                            1.4 µs  22.2%      1
total                                                              6.5 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.9 µs  93.3%      1
_finish                                                            1.3 µs   6.7%      1
total                                                             19.1 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                            2.7 µs  68.3%      1
_finish                                                            1.2 µs  31.7%      1
total                                                              3.9 µs
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
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    13.2 µs  80.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs  19.3%      1
total                                                             16.3 µs
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
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.7 µs 100.0%      1
total                                                             24.7 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           19.9 µs 100.0%      1
total                                                             19.9 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.7 µs  86.9%      1
_finish                                                            1.3 µs  13.1%      1
total                                                             10.0 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           19.7 µs 100.0%      1
total                                                             19.7 µs
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
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           27.9 µs 100.0%      1
total                                                             27.9 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          25.9 µs 100.0%      1
total                                                             25.9 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           14.2 µs  90.8%      1
_finish                                                            1.4 µs   9.2%      1
total                                                             15.6 µs
```

