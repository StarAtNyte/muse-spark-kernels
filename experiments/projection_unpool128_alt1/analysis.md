# projection_unpool128_alt1

Job: `fa6256f742b54390`
Correctness: True; official speedup: 1.52999x; CI: [1.5070366209355686, 1.5529362134513727]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.19 | 9.49 | 1.495x | 7.87 |
| decode 1 | o | 6.70 | 4.87 | 1.376x | 3.50 |
| decode 1 | pooler | 21.30 | 13.58 | 1.568x | 13.99 |
| decode 1 | unpooler | 6.78 | 3.64 | 1.863x | 1.75 |
| decode 1 | router | 10.61 | 3.99 | 2.661x | 1.75 |
| decode 16 | qkv_gate | 14.02 | 9.87 | 1.419x | 7.93 |
| decode 16 | o | 6.50 | 4.98 | 1.304x | 3.56 |
| decode 16 | pooler | 21.42 | 13.67 | 1.567x | 14.06 |
| decode 16 | unpooler | 18.72 | 13.53 | 1.383x | 14.06 |
| decode 16 | router | 15.01 | 4.02 | 3.734x | 1.81 |
| decode 32 | qkv_gate | 14.61 | 11.95 | 1.223x | 8.00 |
| decode 32 | o | 6.78 | 5.75 | 1.180x | 3.62 |
| decode 32 | pooler | 21.89 | 16.43 | 1.332x | 14.14 |
| decode 32 | unpooler | 18.75 | 15.56 | 1.205x | 14.14 |
| decode 32 | router | 15.06 | 5.04 | 2.989x | 1.86 |
| verify 1 | qkv_gate | 14.21 | 9.45 | 1.504x | 7.90 |
| verify 1 | o | 6.66 | 4.99 | 1.333x | 3.53 |
| verify 1 | pooler | 21.58 | 13.85 | 1.559x | 14.02 |
| verify 1 | unpooler | 6.98 | 4.18 | 1.668x | 1.76 |
| verify 1 | router | 26.21 | 4.17 | 6.284x | 1.78 |
| verify 16 | qkv_gate | 17.02 | 14.20 | 1.199x | 8.42 |
| verify 16 | o | 7.36 | 6.63 | 1.110x | 3.99 |
| verify 16 | pooler | 25.74 | 20.10 | 1.281x | 14.64 |
| verify 16 | unpooler | 20.21 | 18.90 | 1.069x | 14.64 |
| verify 16 | router | 16.18 | 6.11 | 2.645x | 2.21 |
| verify 32 | qkv_gate | 19.70 | 17.39 | 1.132x | 9.77 |
| verify 32 | o | 9.25 | 8.70 | 1.063x | 4.48 |
| verify 32 | pooler | 29.47 | 24.55 | 1.200x | 17.37 |
| verify 32 | unpooler | 24.51 | 23.35 | 1.050x | 17.37 |
| verify 32 | router | 17.52 | 8.14 | 2.152x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.694 | 2.206 | 1.675x | 2.065x |
| decode 16 | 4.691 | 2.857 | 1.642x | 1.827x |
| decode 32 | 4.779 | 3.393 | 1.409x | 1.845x |
| verify 1 | 4.689 | 2.272 | 2.064x | 2.609x |
| verify 16 | 5.364 | 4.089 | 1.312x | 1.971x |
| verify 32 | 6.228 | 5.093 | 1.223x | 1.944x |

Recomputed geometric mean across loads: 1.529986x.
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
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       3.8 µs 100.0%      1
total                                                              3.8 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.7 µs 100.0%      1
total                                                             13.7 µs
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
void rr_scalar_kernel<1, 512, 2>(float const*, __nv_bfloa...       2.8 µs 100.0%      1
total                                                              2.8 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs 100.0%      1
total                                                             10.1 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.2 µs 100.0%      1
total                                                              5.2 µs
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
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.2 µs 100.0%      1
total                                                             13.2 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
void routertail_small_kernel<2, 16, 8>(int4 const*, __nv_...       3.6 µs 100.0%      1
total                                                              3.6 µs
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
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.6 µs 100.0%      1
total                                                              5.6 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.2 µs 100.0%      1
total                                                             16.2 µs
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
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       8.9 µs 100.0%      1
total                                                              8.9 µs
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
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.3 µs 100.0%      1
total                                                             13.3 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       3.3 µs 100.0%      1
total                                                              3.3 µs
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
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.4 µs 100.0%      1
total                                                             13.4 µs
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
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     20.4 µs 100.0%      1
total                                                             20.4 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_coopB_NNN                     18.8 µs 100.0%      1
total                                                             18.8 µs
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
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           18.4 µs 100.0%      1
total                                                             18.4 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.5 µs 100.0%      1
total                                                              8.5 µs
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
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    24.0 µs 100.0%      1
total                                                             24.0 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.6 µs 100.0%      1
total                                                              7.6 µs
```

