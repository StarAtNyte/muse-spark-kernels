# projection_nathen_v18

Job: `575f2c20ec7f4236`
Correctness: True; official speedup: 1.51055x; CI: [1.4878929380570232, 1.5332094742415008]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 14.16 | 9.52 | 1.487x | 7.87 |
| decode 1 | o | 6.77 | 5.00 | 1.353x | 3.50 |
| decode 1 | pooler | 21.78 | 13.87 | 1.570x | 13.99 |
| decode 1 | unpooler | 7.04 | 4.05 | 1.738x | 1.75 |
| decode 1 | router | 10.56 | 4.11 | 2.571x | 1.75 |
| decode 16 | qkv_gate | 14.24 | 10.22 | 1.394x | 7.93 |
| decode 16 | o | 6.78 | 5.64 | 1.204x | 3.56 |
| decode 16 | pooler | 21.52 | 13.80 | 1.559x | 14.06 |
| decode 16 | unpooler | 18.42 | 13.54 | 1.360x | 14.06 |
| decode 16 | router | 15.07 | 4.66 | 3.235x | 1.81 |
| decode 32 | qkv_gate | 14.21 | 11.87 | 1.197x | 8.00 |
| decode 32 | o | 7.10 | 6.10 | 1.164x | 3.62 |
| decode 32 | pooler | 22.18 | 16.51 | 1.343x | 14.14 |
| decode 32 | unpooler | 18.85 | 15.73 | 1.198x | 14.14 |
| decode 32 | router | 15.46 | 5.48 | 2.820x | 1.86 |
| verify 1 | qkv_gate | 13.94 | 9.23 | 1.511x | 7.90 |
| verify 1 | o | 6.64 | 5.35 | 1.240x | 3.53 |
| verify 1 | pooler | 21.31 | 13.64 | 1.562x | 14.02 |
| verify 1 | unpooler | 6.94 | 4.34 | 1.600x | 1.76 |
| verify 1 | router | 26.24 | 4.34 | 6.052x | 1.78 |
| verify 16 | qkv_gate | 17.01 | 13.88 | 1.225x | 8.42 |
| verify 16 | o | 7.20 | 6.77 | 1.064x | 3.99 |
| verify 16 | pooler | 24.90 | 20.22 | 1.231x | 14.64 |
| verify 16 | unpooler | 19.84 | 19.11 | 1.038x | 14.64 |
| verify 16 | router | 16.61 | 6.63 | 2.504x | 2.21 |
| verify 32 | qkv_gate | 20.08 | 17.24 | 1.165x | 9.77 |
| verify 32 | o | 9.44 | 9.25 | 1.020x | 4.48 |
| verify 32 | pooler | 31.23 | 24.53 | 1.273x | 17.37 |
| verify 32 | unpooler | 24.42 | 23.43 | 1.042x | 17.37 |
| verify 32 | router | 17.92 | 8.57 | 2.092x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.739 | 2.266 | 1.650x | 2.090x |
| decode 16 | 4.714 | 2.967 | 1.589x | 1.836x |
| decode 32 | 4.823 | 3.453 | 1.397x | 1.862x |
| verify 1 | 4.654 | 2.288 | 2.035x | 2.590x |
| verify 16 | 5.304 | 4.129 | 1.284x | 1.949x |
| verify 32 | 6.391 | 5.147 | 1.242x | 1.995x |

Recomputed geometric mean across loads: 1.510551x.
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
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       4.0 µs 100.0%      1
total                                                              4.0 µs
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
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.4 µs 100.0%      1
total                                                              3.4 µs
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
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.5 µs 100.0%      1
total                                                             10.5 µs
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
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      13.1 µs 100.0%      1
total                                                             13.1 µs
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
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.8 µs 100.0%      1
total                                                             11.8 µs
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
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.8 µs 100.0%      1
total                                                             16.8 µs
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
void rr_kernel<32, 16, 4, 1, 2, 1>(CUtensorMap_st, CUtens...       4.4 µs 100.0%      1
total                                                              4.4 µs
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
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       4.6 µs 100.0%      1
total                                                              4.6 µs
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
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.5 µs 100.0%      1
total                                                             13.5 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.8 µs 100.0%      1
total                                                              6.8 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     21.7 µs 100.0%      1
total                                                             21.7 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.1 µs 100.0%      1
total                                                             19.1 µs
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
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           17.6 µs 100.0%      1
total                                                             17.6 µs
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
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           24.2 µs 100.0%      1
total                                                             24.2 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.4 µs 100.0%      1
total                                                             23.4 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
void rwdl_256_kernel<64, 4, 8, 0, 2, 3, 1, 0>(CUtensorMap...       7.7 µs 100.0%      1
total                                                              7.7 µs
```

