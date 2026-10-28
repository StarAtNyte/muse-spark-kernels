# projection_128_lt_stack

Job: `47d6734c1cfb4dca`
Correctness: True; official speedup: 1.52362x; CI: [1.5007657787053625, 1.5464743811024801]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.76 | 9.26 | 1.485x | 7.87 |
| decode 1 | o | 7.01 | 5.26 | 1.333x | 3.50 |
| decode 1 | pooler | 21.07 | 13.47 | 1.564x | 13.99 |
| decode 1 | unpooler | 6.94 | 3.96 | 1.755x | 1.75 |
| decode 1 | router | 10.56 | 4.11 | 2.569x | 1.75 |
| decode 16 | qkv_gate | 14.08 | 10.30 | 1.367x | 7.93 |
| decode 16 | o | 6.98 | 5.69 | 1.227x | 3.56 |
| decode 16 | pooler | 21.44 | 13.84 | 1.549x | 14.06 |
| decode 16 | unpooler | 18.59 | 13.61 | 1.366x | 14.06 |
| decode 16 | router | 14.82 | 4.44 | 3.340x | 1.81 |
| decode 32 | qkv_gate | 14.67 | 12.10 | 1.213x | 8.00 |
| decode 32 | o | 6.91 | 5.94 | 1.164x | 3.62 |
| decode 32 | pooler | 21.82 | 16.27 | 1.342x | 14.14 |
| decode 32 | unpooler | 18.88 | 15.84 | 1.192x | 14.14 |
| decode 32 | router | 15.42 | 5.27 | 2.927x | 1.86 |
| verify 1 | qkv_gate | 13.84 | 9.07 | 1.526x | 7.90 |
| verify 1 | o | 6.77 | 5.30 | 1.276x | 3.53 |
| verify 1 | pooler | 21.41 | 13.47 | 1.589x | 14.02 |
| verify 1 | unpooler | 7.17 | 4.38 | 1.636x | 1.76 |
| verify 1 | router | 26.02 | 4.00 | 6.504x | 1.78 |
| verify 16 | qkv_gate | 17.25 | 14.02 | 1.230x | 8.42 |
| verify 16 | o | 7.71 | 6.92 | 1.114x | 3.99 |
| verify 16 | pooler | 25.09 | 20.78 | 1.207x | 14.64 |
| verify 16 | unpooler | 19.90 | 18.57 | 1.072x | 14.64 |
| verify 16 | router | 16.51 | 6.27 | 2.632x | 2.21 |
| verify 32 | qkv_gate | 20.08 | 17.09 | 1.175x | 9.77 |
| verify 32 | o | 9.44 | 8.85 | 1.067x | 4.48 |
| verify 32 | pooler | 30.91 | 23.80 | 1.299x | 17.37 |
| verify 32 | unpooler | 24.35 | 23.01 | 1.059x | 17.37 |
| verify 32 | router | 17.31 | 7.78 | 2.226x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.679 | 2.236 | 1.646x | 2.057x |
| decode 16 | 4.706 | 2.968 | 1.586x | 1.832x |
| decode 32 | 4.818 | 3.436 | 1.402x | 1.860x |
| verify 1 | 4.662 | 2.246 | 2.076x | 2.594x |
| verify 16 | 5.361 | 4.127 | 1.299x | 1.970x |
| verify 32 | 6.330 | 4.992 | 1.268x | 1.976x |

Recomputed geometric mean across loads: 1.523620x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.6 µs 100.0%      1
total                                                              9.6 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       3.9 µs 100.0%      1
total                                                              3.9 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.8 µs 100.0%      1
total                                                             13.8 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.3 µs 100.0%      1
total                                                              3.3 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
void rr_scalar_kernel<1, 512, 2>(float const*, __nv_bfloa...       2.8 µs 100.0%      1
total                                                              2.8 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.2 µs 100.0%      1
total                                                             10.2 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs 100.0%      1
total                                                              5.6 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      13.7 µs 100.0%      1
total                                                             13.7 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      12.9 µs 100.0%      1
total                                                             12.9 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
void routertail_small_kernel<2, 16, 8>(int4 const*, __nv_...       3.9 µs 100.0%      1
total                                                              3.9 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.5 µs 100.0%      1
total                                                             11.5 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.8 µs 100.0%      1
total                                                              5.8 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.6 µs 100.0%      1
total                                                             16.6 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.1 µs 100.0%      1
total                                                             15.1 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
void rr_kernel<32, 16, 4, 1, 2, 1>(CUtensorMap_st, CUtens...       4.1 µs 100.0%      1
total                                                              4.1 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.1 µs 100.0%      1
total                                                              9.1 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       4.4 µs 100.0%      1
total                                                              4.4 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.2 µs 100.0%      1
total                                                             13.2 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       3.4 µs 100.0%      1
total                                                              3.4 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
void routertail_small_kernel<1, 16, 8>(int4 const*, __nv_...       3.3 µs 100.0%      1
total                                                              3.3 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.9 µs 100.0%      1
total                                                             13.9 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.7 µs 100.0%      1
total                                                              6.7 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_NNT                           20.6 µs 100.0%      1
total                                                             20.6 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN                     18.2 µs 100.0%      1
total                                                             18.2 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
void rwdl_128_kernel<64, 8, 8, 1, 2, 1>(CUtensorMap_st, C...       5.8 µs 100.0%      1
total                                                              5.8 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           18.8 µs 100.0%      1
total                                                             18.8 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.4 µs 100.0%      1
total                                                              8.4 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           24.0 µs 100.0%      1
total                                                             24.0 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.6 µs 100.0%      1
total                                                             23.6 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.6 µs 100.0%      1
total                                                              7.6 µs
```

