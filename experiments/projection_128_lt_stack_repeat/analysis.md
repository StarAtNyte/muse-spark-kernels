# projection_128_lt_stack_repeat

Job: `f4dc4cc10dcd4e5a`
Correctness: True; official speedup: 1.51860x; CI: [1.4958253639778845, 1.5413834968909164]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.11 | 9.46 | 1.492x | 7.87 |
| decode 1 | o | 7.04 | 5.39 | 1.305x | 3.50 |
| decode 1 | pooler | 21.44 | 13.58 | 1.579x | 13.99 |
| decode 1 | unpooler | 6.72 | 3.85 | 1.745x | 1.75 |
| decode 1 | router | 10.40 | 3.86 | 2.694x | 1.75 |
| decode 16 | qkv_gate | 14.27 | 10.13 | 1.408x | 7.93 |
| decode 16 | o | 7.01 | 5.73 | 1.224x | 3.56 |
| decode 16 | pooler | 21.57 | 13.80 | 1.563x | 14.06 |
| decode 16 | unpooler | 18.85 | 14.06 | 1.340x | 14.06 |
| decode 16 | router | 14.50 | 4.12 | 3.516x | 1.81 |
| decode 32 | qkv_gate | 14.54 | 11.95 | 1.217x | 8.00 |
| decode 32 | o | 6.78 | 6.00 | 1.131x | 3.62 |
| decode 32 | pooler | 21.76 | 16.26 | 1.338x | 14.14 |
| decode 32 | unpooler | 19.07 | 15.99 | 1.193x | 14.14 |
| decode 32 | router | 15.76 | 5.61 | 2.809x | 1.86 |
| verify 1 | qkv_gate | 14.78 | 9.60 | 1.540x | 7.90 |
| verify 1 | o | 6.56 | 4.75 | 1.380x | 3.53 |
| verify 1 | pooler | 21.63 | 13.62 | 1.589x | 14.02 |
| verify 1 | unpooler | 7.07 | 4.36 | 1.622x | 1.76 |
| verify 1 | router | 26.51 | 4.31 | 6.158x | 1.78 |
| verify 16 | qkv_gate | 17.06 | 13.69 | 1.245x | 8.42 |
| verify 16 | o | 7.50 | 6.81 | 1.103x | 3.99 |
| verify 16 | pooler | 24.94 | 21.14 | 1.180x | 14.64 |
| verify 16 | unpooler | 20.16 | 19.21 | 1.049x | 14.64 |
| verify 16 | router | 16.32 | 6.20 | 2.634x | 2.21 |
| verify 32 | qkv_gate | 20.02 | 17.53 | 1.142x | 9.77 |
| verify 32 | o | 9.22 | 8.72 | 1.057x | 4.48 |
| verify 32 | pooler | 30.74 | 24.20 | 1.270x | 17.37 |
| verify 32 | unpooler | 24.54 | 23.29 | 1.054x | 17.37 |
| verify 32 | router | 17.41 | 8.10 | 2.149x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.702 | 2.241 | 1.652x | 2.069x |
| decode 16 | 4.724 | 2.966 | 1.592x | 1.839x |
| decode 32 | 4.831 | 3.460 | 1.396x | 1.865x |
| verify 1 | 4.747 | 2.271 | 2.090x | 2.641x |
| verify 16 | 5.331 | 4.157 | 1.282x | 1.959x |
| verify 32 | 6.319 | 5.074 | 1.245x | 1.973x |

Recomputed geometric mean across loads: 1.518604x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...      10.5 µs 100.0%      1
total                                                             10.5 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       3.8 µs 100.0%      1
total                                                              3.8 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.9 µs 100.0%      1
total                                                             13.9 µs
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
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.5 µs 100.0%      1
total                                                             10.5 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.3 µs 100.0%      1
total                                                              5.3 µs
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
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.3 µs 100.0%      1
total                                                             13.3 µs
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
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.4 µs 100.0%      1
total                                                             16.4 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.3 µs 100.0%      1
total                                                             15.3 µs
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
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.2 µs 100.0%      1
total                                                              9.2 µs
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
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.8 µs 100.0%      1
total                                                             13.8 µs
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
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_NNT                           20.7 µs 100.0%      1
total                                                             20.7 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN                     18.7 µs 100.0%      1
total                                                             18.7 µs
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
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           22.3 µs 100.0%      1
total                                                             22.3 µs
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
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           24.3 µs 100.0%      1
total                                                             24.3 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.9 µs 100.0%      1
total                                                             23.9 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.7 µs 100.0%      1
total                                                              7.7 µs
```

