# projection_lt_stack_v2

Job: `11d610c508004575`
Correctness: True; official speedup: 1.51597x; CI: [1.4932285077173055, 1.5387075485614874]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.22 | 9.80 | 1.452x | 7.87 |
| decode 1 | o | 6.83 | 5.24 | 1.305x | 3.50 |
| decode 1 | pooler | 21.31 | 13.73 | 1.552x | 13.99 |
| decode 1 | unpooler | 6.45 | 3.67 | 1.756x | 1.75 |
| decode 1 | router | 10.21 | 3.79 | 2.694x | 1.75 |
| decode 16 | qkv_gate | 13.86 | 10.04 | 1.380x | 7.93 |
| decode 16 | o | 7.06 | 5.90 | 1.196x | 3.56 |
| decode 16 | pooler | 21.70 | 14.10 | 1.539x | 14.06 |
| decode 16 | unpooler | 18.77 | 13.68 | 1.372x | 14.06 |
| decode 16 | router | 14.93 | 4.40 | 3.390x | 1.81 |
| decode 32 | qkv_gate | 14.21 | 11.69 | 1.215x | 8.00 |
| decode 32 | o | 6.62 | 5.64 | 1.175x | 3.62 |
| decode 32 | pooler | 21.57 | 15.93 | 1.354x | 14.14 |
| decode 32 | unpooler | 18.82 | 15.76 | 1.194x | 14.14 |
| decode 32 | router | 14.88 | 4.89 | 3.041x | 1.86 |
| verify 1 | qkv_gate | 14.30 | 9.55 | 1.497x | 7.90 |
| verify 1 | o | 6.72 | 5.25 | 1.280x | 3.53 |
| verify 1 | pooler | 21.23 | 13.29 | 1.597x | 14.02 |
| verify 1 | unpooler | 7.20 | 4.34 | 1.658x | 1.76 |
| verify 1 | router | 26.22 | 4.21 | 6.230x | 1.78 |
| verify 16 | qkv_gate | 17.50 | 13.80 | 1.268x | 8.42 |
| verify 16 | o | 7.57 | 6.83 | 1.108x | 3.99 |
| verify 16 | pooler | 24.74 | 20.94 | 1.181x | 14.64 |
| verify 16 | unpooler | 19.90 | 18.54 | 1.074x | 14.64 |
| verify 16 | router | 16.38 | 6.22 | 2.634x | 2.21 |
| verify 32 | qkv_gate | 19.84 | 17.97 | 1.104x | 9.77 |
| verify 32 | o | 9.57 | 9.00 | 1.063x | 4.48 |
| verify 32 | pooler | 30.98 | 24.41 | 1.269x | 17.37 |
| verify 32 | unpooler | 24.50 | 23.04 | 1.063x | 17.37 |
| verify 32 | router | 17.50 | 8.03 | 2.179x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.659 | 2.246 | 1.629x | 2.046x |
| decode 16 | 4.731 | 2.983 | 1.586x | 1.842x |
| decode 32 | 4.718 | 3.343 | 1.411x | 1.822x |
| verify 1 | 4.692 | 2.272 | 2.065x | 2.611x |
| verify 16 | 5.338 | 4.113 | 1.298x | 1.961x |
| verify 32 | 6.348 | 5.112 | 1.242x | 1.982x |

Recomputed geometric mean across loads: 1.515968x.
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
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       3.9 µs 100.0%      1
total                                                              3.9 µs
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
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.6 µs 100.0%      1
total                                                             10.6 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.4 µs 100.0%      1
total                                                              5.4 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.0 µs 100.0%      1
total                                                             14.0 µs
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
void routertail_small_kernel<2, 16, 8>(int4 const*, __nv_...       3.8 µs 100.0%      1
total                                                              3.8 µs
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
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.8 µs 100.0%      1
total                                                             17.8 µs
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
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       4.5 µs 100.0%      1
total                                                              4.5 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.4 µs 100.0%      1
total                                                             13.4 µs
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
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs 100.0%      1
total                                                             13.6 µs
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
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_NNT                           21.3 µs 100.0%      1
total                                                             21.3 µs
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
void rwdl_128_kernel<64, 8, 8, 1, 2, 1>(CUtensorMap_st, C...       5.7 µs 100.0%      1
total                                                              5.7 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_80x64_64x11_1x1_v_bz_TNN                           18.3 µs 100.0%      1
total                                                             18.3 µs
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
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           24.7 µs 100.0%      1
total                                                             24.7 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.3 µs 100.0%      1
total                                                             23.3 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.6 µs 100.0%      1
total                                                              7.6 µs
```

