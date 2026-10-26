# bf16_wgmma_v6

Job: `f791d84962934074`
Correctness: True; official speedup: 1.06846x; CI: [1.0524312177222324, 1.0844849603939755]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.19 | 12.47 | 1.138x | 7.87 |
| decode 1 | o | 6.91 | 6.00 | 1.153x | 3.50 |
| decode 1 | pooler | 21.09 | 18.04 | 1.169x | 13.99 |
| decode 1 | unpooler | 7.17 | 5.99 | 1.196x | 1.75 |
| decode 1 | router | 10.37 | 6.67 | 1.555x | 1.75 |
| decode 16 | qkv_gate | 14.08 | 13.69 | 1.028x | 7.93 |
| decode 16 | o | 6.91 | 6.86 | 1.008x | 3.56 |
| decode 16 | pooler | 21.39 | 19.34 | 1.106x | 14.06 |
| decode 16 | unpooler | 18.53 | 18.27 | 1.014x | 14.06 |
| decode 16 | router | 15.14 | 7.19 | 2.106x | 1.81 |
| decode 32 | qkv_gate | 14.26 | 13.96 | 1.021x | 8.00 |
| decode 32 | o | 7.10 | 6.94 | 1.023x | 3.62 |
| decode 32 | pooler | 21.73 | 21.45 | 1.013x | 14.14 |
| decode 32 | unpooler | 18.91 | 18.78 | 1.007x | 14.14 |
| decode 32 | router | 15.46 | 7.42 | 2.083x | 1.86 |
| verify 1 | qkv_gate | 14.40 | 13.88 | 1.038x | 7.90 |
| verify 1 | o | 6.85 | 6.74 | 1.017x | 3.53 |
| verify 1 | pooler | 21.14 | 19.10 | 1.106x | 14.02 |
| verify 1 | unpooler | 7.25 | 7.13 | 1.017x | 1.76 |
| verify 1 | router | 26.02 | 6.81 | 3.818x | 1.78 |
| verify 16 | qkv_gate | 17.20 | 23.75 | 0.724x | 8.42 |
| verify 16 | o | 7.52 | 8.20 | 0.917x | 3.99 |
| verify 16 | pooler | 25.17 | 25.79 | 0.976x | 14.64 |
| verify 16 | unpooler | 19.84 | 22.90 | 0.867x | 14.64 |
| verify 16 | router | 16.54 | 9.87 | 1.676x | 2.21 |
| verify 32 | qkv_gate | 19.73 | 38.20 | 0.516x | 9.77 |
| verify 32 | o | 9.49 | 14.14 | 0.671x | 4.48 |
| verify 32 | pooler | 28.80 | 37.85 | 0.761x | 17.37 |
| verify 32 | unpooler | 24.05 | 40.83 | 0.589x | 17.37 |
| verify 32 | router | 17.76 | 13.00 | 1.366x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.703 | 3.049 | 1.215x | 2.070x |
| decode 16 | 4.715 | 4.052 | 1.164x | 1.836x |
| decode 32 | 4.802 | 4.250 | 1.130x | 1.854x |
| verify 1 | 4.690 | 3.327 | 1.410x | 2.610x |
| verify 16 | 5.349 | 5.611 | 0.953x | 1.965x |
| verify 32 | 6.189 | 8.929 | 0.693x | 1.932x |

Recomputed geometric mean across loads: 1.068458x.
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
_gemv                                                              4.2 µs 100.0%      1
total                                                              4.2 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
_gemv                                                             17.7 µs 100.0%      1
total                                                             17.7 µs
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
_matmul                                                            3.6 µs  70.4%      1
_finish                                                            1.5 µs  29.6%      1
total                                                              5.1 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                     7.8 µs  82.4%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  17.6%      1
total                                                              9.5 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                            4.7 µs 100.0%      1
total                                                              4.7 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.9 µs  94.1%      1
_finish                                                            1.1 µs   5.9%      1
total                                                             19.0 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           18.2 µs 100.0%      1
total                                                             18.2 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.5 µs  68.7%      1
_finish                                                            1.6 µs  31.3%      1
total                                                              5.1 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                     8.6 µs  82.5%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.8 µs  17.5%      1
total                                                             10.4 µs
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
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   19.3 µs  90.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       2.0 µs   9.3%      1
total                                                             21.3 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           18.9 µs 100.0%      1
total                                                             18.9 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  69.1%      1
_finish                                                            1.7 µs  30.9%      1
total                                                              5.4 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                     8.6 µs  83.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.7 µs  16.8%      1
total                                                             10.3 µs
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
_matmul                                                           17.4 µs  94.4%      1
_finish                                                            1.0 µs   5.6%      1
total                                                             18.4 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      4.0 µs  72.0%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       1.6 µs  28.0%      1
total                                                              5.6 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  70.5%      1
_finish                                                            1.5 µs  29.5%      1
total                                                              5.1 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           21.5 µs  92.4%      1
_finish                                                            1.8 µs   7.6%      1
total                                                             23.3 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.4 µs 100.0%      1
total                                                              6.4 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           24.5 µs  92.8%      1
_finish                                                            1.9 µs   7.2%      1
total                                                             26.4 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           23.9 µs 100.0%      1
total                                                             23.9 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.3 µs  78.0%      1
_finish                                                            1.8 µs  22.0%      1
total                                                              8.1 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           36.1 µs  94.9%      1
_finish                                                            2.0 µs   5.1%      1
total                                                             38.1 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                           11.9 µs 100.0%      1
total                                                             11.9 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           38.5 µs 100.0%      1
total                                                             38.5 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           42.4 µs 100.0%      1
total                                                             42.4 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.1 µs  87.8%      1
_finish                                                            1.4 µs  12.2%      1
total                                                             11.5 µs
```

