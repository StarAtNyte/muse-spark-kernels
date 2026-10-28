# projection_unpool128_alt1_repeat

Job: `dab8ec860dae447d`
Correctness: True; official speedup: 1.50998x; CI: [1.4873300744331037, 1.5326294675630459]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.30 | 9.92 | 1.442x | 7.87 |
| decode 1 | o | 6.66 | 4.93 | 1.351x | 3.50 |
| decode 1 | pooler | 21.76 | 13.75 | 1.582x | 13.99 |
| decode 1 | unpooler | 7.17 | 4.18 | 1.714x | 1.75 |
| decode 1 | router | 10.53 | 3.96 | 2.661x | 1.75 |
| decode 16 | qkv_gate | 14.30 | 10.43 | 1.371x | 7.93 |
| decode 16 | o | 6.53 | 5.18 | 1.259x | 3.56 |
| decode 16 | pooler | 21.86 | 14.57 | 1.500x | 14.06 |
| decode 16 | unpooler | 18.21 | 13.44 | 1.355x | 14.06 |
| decode 16 | router | 14.59 | 4.43 | 3.295x | 1.81 |
| decode 32 | qkv_gate | 14.40 | 12.08 | 1.192x | 8.00 |
| decode 32 | o | 6.75 | 6.00 | 1.125x | 3.62 |
| decode 32 | pooler | 22.05 | 16.81 | 1.312x | 14.14 |
| decode 32 | unpooler | 18.35 | 15.58 | 1.178x | 14.14 |
| decode 32 | router | 15.07 | 5.12 | 2.943x | 1.86 |
| verify 1 | qkv_gate | 14.16 | 9.37 | 1.511x | 7.90 |
| verify 1 | o | 6.56 | 5.07 | 1.293x | 3.53 |
| verify 1 | pooler | 21.54 | 13.78 | 1.562x | 14.02 |
| verify 1 | unpooler | 6.94 | 4.26 | 1.631x | 1.76 |
| verify 1 | router | 26.34 | 4.39 | 5.993x | 1.78 |
| verify 16 | qkv_gate | 17.06 | 13.88 | 1.229x | 8.42 |
| verify 16 | o | 7.17 | 6.51 | 1.101x | 3.99 |
| verify 16 | pooler | 25.15 | 19.87 | 1.266x | 14.64 |
| verify 16 | unpooler | 19.81 | 18.73 | 1.057x | 14.64 |
| verify 16 | router | 16.38 | 6.17 | 2.657x | 2.21 |
| verify 32 | qkv_gate | 20.21 | 18.10 | 1.117x | 9.77 |
| verify 32 | o | 9.55 | 9.10 | 1.049x | 4.48 |
| verify 32 | pooler | 30.85 | 24.29 | 1.270x | 17.37 |
| verify 32 | unpooler | 24.62 | 23.51 | 1.048x | 17.37 |
| verify 32 | router | 16.99 | 7.63 | 2.228x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.746 | 2.278 | 1.644x | 2.094x |
| decode 16 | 4.680 | 2.979 | 1.571x | 1.822x |
| decode 32 | 4.751 | 3.446 | 1.379x | 1.834x |
| verify 1 | 4.683 | 2.286 | 2.048x | 2.606x |
| verify 16 | 5.305 | 4.040 | 1.313x | 1.949x |
| verify 32 | 6.338 | 5.122 | 1.237x | 1.979x |

Recomputed geometric mean across loads: 1.509980x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...      10.0 µs 100.0%      1
total                                                             10.0 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       3.6 µs 100.0%      1
total                                                              3.6 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      14.5 µs 100.0%      1
total                                                             14.5 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.5 µs 100.0%      1
total                                                              3.5 µs
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
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      11.2 µs 100.0%      1
total                                                             11.2 µs
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
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      13.5 µs 100.0%      1
total                                                             13.5 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.0 µs 100.0%      1
total                                                             13.0 µs
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
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.6 µs 100.0%      1
total                                                             11.6 µs
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
void rr_kernel<32, 16, 4, 1, 2, 1>(CUtensorMap_st, CUtens...       4.3 µs 100.0%      1
total                                                              4.3 µs
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
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       4.4 µs 100.0%      1
total                                                              4.4 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.7 µs 100.0%      1
total                                                             13.7 µs
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
void routertail_small_kernel<1, 16, 8>(int4 const*, __nv_...       3.4 µs 100.0%      1
total                                                              3.4 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs 100.0%      1
total                                                             13.6 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.4 µs 100.0%      1
total                                                              6.4 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     20.0 µs 100.0%      1
total                                                             20.0 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN                     18.4 µs 100.0%      1
total                                                             18.4 µs
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
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.1 µs 100.0%      1
total                                                             20.1 µs
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
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           24.5 µs 100.0%      1
total                                                             24.5 µs
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

