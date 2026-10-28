# decoder_stack_v3

Job: `bf77d2d63ce54a6a`
Correctness: True; official speedup: 1.70772x; CI: [1.682106986856607, 1.7333386717354884]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 170.50 | 88.31 | 1.931x | 40.02 |
| decode 1 | full | 181.20 | 97.01 | 1.868x | 44.18 |
| decode 16 | sliding | 267.74 | 157.16 | 1.704x | 76.72 |
| decode 16 | full | 427.47 | 234.36 | 1.824x | 137.46 |
| decode 32 | sliding | 263.65 | 193.67 | 1.361x | 84.51 |
| decode 32 | full | 617.33 | 362.21 | 1.704x | 208.10 |
| verify 1 | sliding | 260.96 | 139.82 | 1.866x | 62.29 |
| verify 1 | full | 269.01 | 147.43 | 1.825x | 66.44 |
| verify 16 | sliding | 534.74 | 322.47 | 1.658x | 126.61 |
| verify 16 | full | 689.70 | 447.12 | 1.543x | 188.68 |
| verify 32 | sliding | 725.10 | 422.23 | 1.717x | 143.81 |
| verify 32 | full | 1070.62 | 696.77 | 1.537x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.742 | 5.614 | 1.913x | 4.216x |
| decode 16 | 19.156 | 10.979 | 1.745x | 3.344x |
| decode 32 | 22.005 | 14.704 | 1.497x | 3.049x |
| verify 1 | 16.308 | 8.791 | 1.855x | 4.152x |
| verify 16 | 35.633 | 21.988 | 1.621x | 4.029x |
| verify 32 | 50.485 | 30.571 | 1.651x | 4.636x |

Recomputed geometric mean across loads: 1.707723x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      20.3 µs  20.1%      1
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.8 µs  13.7%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      13.8 µs  13.6%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.3 µs   9.2%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.2 µs   5.1%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.1 µs   5.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   4.6%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.1 µs   4.0%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.0 µs   3.9%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   3.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.5%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.7%      1
... 8 more
total                                                            101.1 µs
GPU busy (kernels overlap: PDL or streams)                        97.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.7 µs  14.4%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      13.1 µs  13.8%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.9 µs  12.5%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.5 µs  10.0%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       9.1 µs   9.5%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.0 µs   5.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   4.7%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.0 µs   4.2%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.0 µs   4.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   3.9%      1
_post_expert_norm_weighted_sum_kernel                              2.9 µs   3.1%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.8%      1
... 8 more
total                                                             95.2 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      45.3 µs  29.9%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      23.7 µs  15.6%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.7 µs   9.7%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.3 µs   9.4%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.3 µs   6.8%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.6 µs   4.3%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs   3.7%      1
_matmul                                                            5.4 µs   3.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              3.9 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.4%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.8%      1
... 7 more
total                                                            151.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                            95.5 µs  37.0%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      58.7 µs  22.7%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      31.9 µs  12.4%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.4 µs   5.6%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.9 µs   5.4%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   3.9%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs   2.2%      1
_matmul                                                            5.1 µs   2.0%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   1.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.4%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.1%      1
... 5 more
total                                                            258.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      63.2 µs  30.0%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      52.7 µs  25.0%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      18.4 µs   8.7%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.6 µs   7.4%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      12.0 µs   5.7%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.8 µs   4.6%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.5 µs   3.1%      1
_matmul                                                            5.3 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   2.1%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.3%      1
... 7 more
total                                                            210.7 µs
GPU busy (kernels overlap: PDL or streams)                       202.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                           186.0 µs  48.5%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      73.6 µs  19.2%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      44.6 µs  11.6%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.9 µs   4.4%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      16.2 µs   4.2%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      12.0 µs   3.1%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.9 µs   1.5%      1
_matmul                                                            5.3 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              4.9 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.9 µs   0.8%      1
... 5 more
total                                                            383.8 µs
GPU busy (kernels overlap: PDL or streams)                       379.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      37.3 µs  26.6%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      26.7 µs  19.0%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.6 µs   9.7%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.5 µs   6.8%      1
_ha__sliding_partials                                              8.6 µs   6.1%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.8 µs   5.6%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   3.4%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   3.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   2.8%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   2.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.9%      1
... 8 more
total                                                            140.5 µs
GPU busy (kernels overlap: PDL or streams)                       135.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      53.1 µs  32.0%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      26.2 µs  15.8%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.0 µs  10.3%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      14.4 µs   8.7%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...      10.3 µs   6.2%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.9 µs   4.8%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   2.8%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   2.4%      1
_post_expert_norm_weighted_sum_kernel                              3.4 µs   2.0%      1
_ha__merge_norm_gate                                               2.8 µs   1.7%      1
... 8 more
total                                                            165.9 µs
GPU busy (kernels overlap: PDL or streams)                       163.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     126.6 µs  34.5%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     123.6 µs  33.6%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     23.5 µs   6.4%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.9 µs   5.4%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      16.4 µs   4.5%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.6 µs   4.0%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.4 µs   2.0%      1
_matmul                                                            7.4 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.2 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   1.3%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   1.1%      1
_ha__prepare_qkv                                                   3.1 µs   0.9%      1
... 6 more
total                                                            367.5 µs
GPU busy (kernels overlap: PDL or streams)                       323.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           153.2 µs  30.9%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     125.6 µs  25.3%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     123.6 µs  24.9%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     21.9 µs   4.4%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.8 µs   4.0%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.7 µs   2.8%      1
_matmul                                                            7.3 µs   1.5%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.6 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.2 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   0.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.1 µs   0.6%      1
... 5 more
total                                                            496.3 µs
GPU busy (kernels overlap: PDL or streams)                       451.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     158.2 µs  37.5%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     107.7 µs  25.6%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.7 µs   6.8%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.4 µs   5.5%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.1 µs   4.8%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      19.9 µs   4.7%      1
_matmul                                                           10.9 µs   2.6%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       9.0 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.7 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.4 µs   1.8%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   1.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.5 µs   1.1%      1
... 9 more
total                                                            421.3 µs
GPU busy (kernels overlap: PDL or streams)                       398.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           304.7 µs  40.9%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     177.3 µs  23.8%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     137.5 µs  18.5%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.1 µs   3.8%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.5 µs   3.2%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.1 µs   2.7%      1
_matmul                                                           10.8 µs   1.5%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.7 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.4 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.6 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              6.1 µs   0.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.1 µs   0.4%      1
... 5 more
total                                                            744.2 µs
GPU busy (kernels overlap: PDL or streams)                       714.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

