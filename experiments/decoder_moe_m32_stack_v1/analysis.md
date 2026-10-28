# decoder_moe_m32_stack_v1

Job: `6749df98f8b64274`
Correctness: True; official speedup: 1.70275x; CI: [1.6772109677601728, 1.7282935353061677]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 171.01 | 87.96 | 1.944x | 40.02 |
| decode 1 | full | 183.26 | 97.23 | 1.885x | 44.18 |
| decode 16 | sliding | 264.37 | 154.83 | 1.707x | 76.72 |
| decode 16 | full | 441.78 | 243.89 | 1.811x | 137.46 |
| decode 32 | sliding | 256.82 | 191.03 | 1.344x | 84.51 |
| decode 32 | full | 615.41 | 360.71 | 1.706x | 208.10 |
| verify 1 | sliding | 268.51 | 145.04 | 1.851x | 62.29 |
| verify 1 | full | 277.12 | 154.64 | 1.792x | 66.44 |
| verify 16 | sliding | 537.70 | 324.11 | 1.659x | 126.61 |
| verify 16 | full | 715.58 | 456.86 | 1.566x | 188.68 |
| verify 32 | sliding | 719.18 | 426.26 | 1.687x | 143.81 |
| verify 32 | full | 1083.14 | 707.19 | 1.532x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.799 | 5.602 | 1.928x | 4.238x |
| decode 16 | 19.229 | 11.024 | 1.744x | 3.357x |
| decode 32 | 21.660 | 14.559 | 1.488x | 3.001x |
| verify 1 | 16.785 | 9.146 | 1.835x | 4.273x |
| verify 16 | 36.183 | 22.219 | 1.628x | 4.092x |
| verify 32 | 50.413 | 30.923 | 1.630x | 4.629x |

Recomputed geometric mean across loads: 1.702752x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.5 µs  15.3%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      13.1 µs  14.9%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.7 µs  11.0%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.7 µs   9.9%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   5.8%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.0 µs   5.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   5.3%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.3 µs   4.9%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   4.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   4.1%      1
_post_expert_norm_weighted_sum_kernel                              2.9 µs   3.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   3.0%      1
... 8 more
total                                                             88.2 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      22.1 µs  19.6%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      17.2 µs  15.3%      1
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.5 µs  12.0%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      12.1 µs  10.7%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.2 µs   8.2%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.2 µs   4.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   3.9%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.9 µs   3.5%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   3.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   3.4%      1
_post_expert_norm_weighted_sum_kernel                              3.4 µs   3.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.4%      1
... 8 more
total                                                            112.6 µs
GPU busy (kernels overlap: PDL or streams)                       105.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      51.4 µs  32.0%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.0 µs  16.8%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      15.0 µs   9.3%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.1 µs   8.8%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   6.3%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.8 µs   4.3%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.5 µs   3.4%      1
_matmul                                                            5.1 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.5%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.7%      1
... 7 more
total                                                            160.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                            95.7 µs  36.8%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      60.7 µs  23.3%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      32.7 µs  12.6%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.3 µs   5.5%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.1 µs   5.4%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   3.9%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.7 µs   2.2%      1
_matmul                                                            5.0 µs   1.9%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.0%      1
... 5 more
total                                                            260.2 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_om__w2                                                          106.2 µs  53.1%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.6 µs   8.8%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.7 µs   7.8%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.6 µs   5.8%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.5 µs   4.8%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.3 µs   3.2%      1
_matmul                                                            5.3 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   2.1%      1
_post_expert_norm_weighted_sum_kernel                              3.9 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.4%      1
_om__route                                                         2.5 µs   1.3%      1
... 6 more
total                                                            200.0 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                           188.4 µs  49.9%      1
_om__w2                                                          109.7 µs  29.1%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.0 µs   4.5%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.7 µs   4.2%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      12.0 µs   3.2%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.9 µs   1.6%      1
_matmul                                                            5.3 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.7%      1
_om__route                                                         2.5 µs   0.7%      1
... 4 more
total                                                            377.6 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      36.5 µs  26.2%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      27.3 µs  19.6%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.6 µs   9.8%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.3 µs   6.7%      1
_ha__sliding_partials                                              8.4 µs   6.0%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.7 µs   5.6%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   3.2%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   2.8%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   2.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   2.0%      1
... 8 more
total                                                            139.4 µs
GPU busy (kernels overlap: PDL or streams)                       134.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      56.0 µs  32.6%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      29.2 µs  17.0%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.2 µs  10.0%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      14.0 µs   8.2%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.3 µs   5.4%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.8 µs   4.6%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   2.7%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   2.5%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   2.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   1.7%      1
... 8 more
total                                                            171.9 µs
GPU busy (kernels overlap: PDL or streams)                       167.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     131.0 µs  34.9%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     130.0 µs  34.6%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.7 µs   6.0%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   5.2%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      16.8 µs   4.5%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.9 µs   3.7%      1
_matmul                                                            7.3 µs   2.0%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.2 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.2 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.3%      1
_post_expert_norm_weighted_sum_kernel                              4.3 µs   1.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.8%      1
... 6 more
total                                                            375.6 µs
GPU busy (kernels overlap: PDL or streams)                       340.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           154.9 µs  30.7%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     137.4 µs  27.3%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     115.1 µs  22.9%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     23.6 µs   4.7%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   3.9%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.7 µs   2.7%      1
_matmul                                                            7.4 µs   1.5%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.6 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.3 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              4.5 µs   0.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.6%      1
... 5 more
total                                                            503.7 µs
GPU busy (kernels overlap: PDL or streams)                       470.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     169.0 µs  37.2%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     129.4 µs  28.5%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.4 µs   6.3%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.2 µs   5.1%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.6 µs   4.5%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.7 µs   4.3%      1
_matmul                                                           10.8 µs   2.4%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.7 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.6 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.3 µs   1.6%      1
_post_expert_norm_weighted_sum_kernel                              5.5 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs   0.9%      1
... 9 more
total                                                            453.9 µs
GPU busy (kernels overlap: PDL or streams)                       430.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           307.6 µs  41.1%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     175.1 µs  23.4%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     126.5 µs  16.9%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.5 µs   3.8%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    24.0 µs   3.2%      1
_matmul                                                           21.6 µs   2.9%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.4 µs   2.7%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       9.1 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.5 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.2 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              6.2 µs   0.8%      1
_pre_rms_kernel                                                    4.1 µs   0.6%      1
... 5 more
total                                                            748.6 µs
GPU busy (kernels overlap: PDL or streams)                       708.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

