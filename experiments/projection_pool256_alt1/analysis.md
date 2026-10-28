# projection_pool256_alt1

Job: `def4b81cba2a4eb6`
Correctness: True; official speedup: 1.52636x; CI: [1.503466614340073, 1.549257475690532]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.32 | 9.54 | 1.502x | 7.87 |
| decode 1 | o | 6.77 | 4.98 | 1.360x | 3.50 |
| decode 1 | pooler | 21.34 | 13.47 | 1.585x | 13.99 |
| decode 1 | unpooler | 7.01 | 3.84 | 1.824x | 1.75 |
| decode 1 | router | 10.21 | 3.48 | 2.936x | 1.75 |
| decode 16 | qkv_gate | 13.70 | 9.86 | 1.389x | 7.93 |
| decode 16 | o | 6.66 | 5.23 | 1.273x | 3.56 |
| decode 16 | pooler | 21.18 | 13.52 | 1.567x | 14.06 |
| decode 16 | unpooler | 18.54 | 13.47 | 1.377x | 14.06 |
| decode 16 | router | 14.64 | 3.94 | 3.716x | 1.81 |
| decode 32 | qkv_gate | 14.06 | 11.67 | 1.205x | 8.00 |
| decode 32 | o | 6.69 | 5.66 | 1.182x | 3.62 |
| decode 32 | pooler | 21.86 | 16.07 | 1.360x | 14.14 |
| decode 32 | unpooler | 18.82 | 15.58 | 1.208x | 14.14 |
| decode 32 | router | 15.02 | 4.98 | 3.020x | 1.86 |
| verify 1 | qkv_gate | 14.34 | 9.43 | 1.520x | 7.90 |
| verify 1 | o | 6.69 | 5.22 | 1.281x | 3.53 |
| verify 1 | pooler | 21.50 | 13.66 | 1.575x | 14.02 |
| verify 1 | unpooler | 7.10 | 4.51 | 1.575x | 1.76 |
| verify 1 | router | 25.82 | 4.01 | 6.441x | 1.78 |
| verify 16 | qkv_gate | 16.96 | 13.64 | 1.243x | 8.42 |
| verify 16 | o | 7.17 | 6.69 | 1.072x | 3.99 |
| verify 16 | pooler | 24.77 | 19.87 | 1.246x | 14.64 |
| verify 16 | unpooler | 19.87 | 19.16 | 1.037x | 14.64 |
| verify 16 | router | 16.51 | 6.54 | 2.525x | 2.21 |
| verify 32 | qkv_gate | 19.86 | 17.17 | 1.156x | 9.77 |
| verify 32 | o | 8.96 | 8.66 | 1.035x | 4.48 |
| verify 32 | pooler | 30.56 | 25.31 | 1.207x | 17.37 |
| verify 32 | unpooler | 24.51 | 23.39 | 1.048x | 17.37 |
| verify 32 | router | 17.60 | 8.19 | 2.149x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.698 | 2.189 | 1.690x | 2.067x |
| decode 16 | 4.633 | 2.853 | 1.624x | 1.804x |
| decode 32 | 4.740 | 3.345 | 1.417x | 1.830x |
| verify 1 | 4.678 | 2.283 | 2.049x | 2.603x |
| verify 16 | 5.287 | 4.086 | 1.294x | 1.943x |
| verify 32 | 6.292 | 5.129 | 1.227x | 1.964x |

Recomputed geometric mean across loads: 1.526362x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.8 µs 100.0%      1
total                                                              9.8 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       4.0 µs 100.0%      1
total                                                              4.0 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      14.1 µs 100.0%      1
total                                                             14.1 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.2 µs 100.0%      1
total                                                              3.2 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
void rr_scalar_kernel<1, 512, 2>(float const*, __nv_bfloa...       2.7 µs 100.0%      1
total                                                              2.7 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.3 µs 100.0%      1
total                                                             10.3 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.5 µs 100.0%      1
total                                                              5.5 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      13.2 µs 100.0%      1
total                                                             13.2 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.5 µs 100.0%      1
total                                                             13.5 µs
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
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.7 µs 100.0%      1
total                                                              5.7 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.3 µs 100.0%      1
total                                                             16.3 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.2 µs 100.0%      1
total                                                             15.2 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
void rr_kernel<32, 16, 4, 1, 2, 1>(CUtensorMap_st, CUtens...       4.2 µs 100.0%      1
total                                                              4.2 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.3 µs 100.0%      1
total                                                              9.3 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       4.7 µs 100.0%      1
total                                                              4.7 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.5 µs 100.0%      1
total                                                             13.5 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       3.5 µs 100.0%      1
total                                                              3.5 µs
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
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.7 µs 100.0%      1
total                                                             13.7 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.6 µs 100.0%      1
total                                                              6.6 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     19.9 µs 100.0%      1
total                                                             19.9 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           18.1 µs 100.0%      1
total                                                             18.1 µs
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
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.2 µs 100.0%      1
total                                                             19.2 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.6 µs 100.0%      1
total                                                              8.6 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_2x1_v_bz_coopB_NNN                     25.6 µs 100.0%      1
total                                                             25.6 µs
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
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.7 µs 100.0%      1
total                                                              7.7 µs
```

