# bf16_splitk_v1

Job: `3f1954adb1044e31`
Correctness: True; official speedup: 1.03575x; CI: [1.0202172762843524, 1.0512898836838758]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.35 | 14.39 | 0.997x | 7.87 |
| decode 1 | o | 6.46 | 7.65 | 0.845x | 3.50 |
| decode 1 | pooler | 21.34 | 19.62 | 1.088x | 13.99 |
| decode 1 | unpooler | 7.12 | 6.98 | 1.020x | 1.75 |
| decode 1 | router | 10.51 | 10.42 | 1.009x | 1.75 |
| decode 16 | qkv_gate | 14.21 | 14.63 | 0.971x | 7.93 |
| decode 16 | o | 6.80 | 8.75 | 0.777x | 3.56 |
| decode 16 | pooler | 21.54 | 20.20 | 1.066x | 14.06 |
| decode 16 | unpooler | 18.72 | 21.20 | 0.883x | 14.06 |
| decode 16 | router | 14.69 | 10.26 | 1.432x | 1.81 |
| decode 32 | qkv_gate | 14.35 | 15.45 | 0.929x | 8.00 |
| decode 32 | o | 6.86 | 11.63 | 0.590x | 3.62 |
| decode 32 | pooler | 22.05 | 21.97 | 1.003x | 14.14 |
| decode 32 | unpooler | 18.53 | 25.42 | 0.729x | 14.14 |
| decode 32 | router | 15.26 | 10.49 | 1.454x | 1.86 |
| verify 1 | qkv_gate | 14.11 | 14.09 | 1.002x | 7.90 |
| verify 1 | o | 6.88 | 8.21 | 0.838x | 3.53 |
| verify 1 | pooler | 21.74 | 20.27 | 1.072x | 14.02 |
| verify 1 | unpooler | 7.14 | 7.13 | 1.001x | 1.76 |
| verify 1 | router | 25.90 | 10.05 | 2.576x | 1.78 |
| verify 16 | qkv_gate | 17.06 | 17.02 | 1.002x | 8.42 |
| verify 16 | o | 7.42 | 7.64 | 0.972x | 3.99 |
| verify 16 | pooler | 25.36 | 25.03 | 1.013x | 14.64 |
| verify 16 | unpooler | 19.62 | 19.80 | 0.991x | 14.64 |
| verify 16 | router | 16.34 | 11.85 | 1.379x | 2.21 |
| verify 32 | qkv_gate | 19.73 | 19.80 | 0.996x | 9.77 |
| verify 32 | o | 9.47 | 9.65 | 0.982x | 4.48 |
| verify 32 | pooler | 30.08 | 31.05 | 0.969x | 17.37 |
| verify 32 | unpooler | 24.26 | 24.35 | 0.996x | 17.37 |
| verify 32 | router | 17.41 | 16.79 | 1.037x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.707 | 3.662 | 1.012x | 2.072x |
| decode 16 | 4.709 | 4.652 | 1.012x | 1.834x |
| decode 32 | 4.777 | 5.268 | 0.907x | 1.845x |
| verify 1 | 4.698 | 3.705 | 1.268x | 2.614x |
| verify 16 | 5.319 | 5.043 | 1.055x | 1.954x |
| verify 32 | 6.259 | 6.301 | 0.993x | 1.954x |

Recomputed geometric mean across loads: 1.035754x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.7 µs  87.7%      1
_finish                                                            1.2 µs  12.3%      1
total                                                              9.9 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            4.8 µs  79.4%      1
_finish                                                            1.2 µs  20.6%      1
total                                                              6.0 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.2 µs  93.4%      1
_finish                                                            1.2 µs   6.6%      1
total                                                             18.5 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                            2.7 µs  69.7%      1
_finish                                                            1.2 µs  30.3%      1
total                                                              3.8 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.2 µs  81.3%      1
_finish                                                            1.2 µs  18.7%      1
total                                                              6.3 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            9.0 µs  87.3%      1
_finish                                                            1.3 µs  12.7%      1
total                                                             10.3 µs
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
_matmul                                                           18.5 µs  92.9%      1
_finish                                                            1.4 µs   7.1%      1
total                                                             19.9 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           20.2 µs  92.4%      1
_finish                                                            1.7 µs   7.6%      1
total                                                             21.8 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.1 µs  80.7%      1
_finish                                                            1.2 µs  19.3%      1
total                                                              6.3 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           12.4 µs  89.3%      1
_finish                                                            1.5 µs  10.7%      1
total                                                             13.8 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            7.7 µs  77.2%      1
_finish                                                            2.3 µs  22.8%      1
total                                                             10.0 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           21.1 µs  92.6%      1
_finish                                                            1.7 µs   7.4%      1
total                                                             22.8 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           23.8 µs  90.4%      1
_finish                                                            2.5 µs   9.6%      1
total                                                             26.3 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.2 µs  80.6%      1
_finish                                                            1.2 µs  19.4%      1
total                                                              6.4 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.3 µs  86.7%      1
_finish                                                            1.3 µs  13.3%      1
total                                                              9.6 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.2 µs  78.5%      1
_finish                                                            1.4 µs  21.5%      1
total                                                              6.6 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.0 µs  93.3%      1
_finish                                                            1.3 µs   6.7%      1
total                                                             19.2 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                            2.8 µs  69.4%      1
_finish                                                            1.2 µs  30.6%      1
total                                                              4.0 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            5.2 µs  81.0%      1
_finish                                                            1.2 µs  19.0%      1
total                                                              6.4 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    12.9 µs  80.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs  19.3%      1
total                                                             15.9 µs
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
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.2 µs 100.0%      1
total                                                             24.2 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           19.8 µs 100.0%      1
total                                                             19.8 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.4 µs  86.5%      1
_finish                                                            1.3 µs  13.5%      1
total                                                              9.7 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           20.2 µs 100.0%      1
total                                                             20.2 µs
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
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           30.9 µs 100.0%      1
total                                                             30.9 µs
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
_matmul                                                           14.2 µs  90.8%      1
_finish                                                            1.4 µs   9.2%      1
total                                                             15.7 µs
```

