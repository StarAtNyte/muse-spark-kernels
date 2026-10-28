# decoder_stack_m128_fullverify4_repeat_v1

Job: `18099c0d621648ff`
Correctness: True; official speedup: 1.74300x; CI: [1.7168553984970305, 1.7691454106339957]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 172.96 | 89.78 | 1.927x | 40.02 |
| decode 1 | full | 182.83 | 98.64 | 1.854x | 44.18 |
| decode 16 | sliding | 267.86 | 159.39 | 1.681x | 76.72 |
| decode 16 | full | 439.44 | 239.22 | 1.837x | 137.46 |
| decode 32 | sliding | 259.17 | 194.16 | 1.335x | 84.51 |
| decode 32 | full | 616.85 | 360.72 | 1.710x | 208.10 |
| verify 1 | sliding | 260.83 | 135.64 | 1.923x | 62.29 |
| verify 1 | full | 282.50 | 156.63 | 1.804x | 66.44 |
| verify 16 | sliding | 555.58 | 317.67 | 1.749x | 126.61 |
| verify 16 | full | 719.92 | 450.07 | 1.600x | 188.68 |
| verify 32 | sliding | 755.55 | 406.75 | 1.858x | 143.81 |
| verify 32 | full | 1098.72 | 665.52 | 1.651x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.881 | 5.708 | 1.906x | 4.271x |
| decode 16 | 19.352 | 11.159 | 1.734x | 3.378x |
| decode 32 | 21.791 | 14.703 | 1.482x | 3.020x |
| verify 1 | 16.518 | 8.745 | 1.889x | 4.205x |
| verify 16 | 37.076 | 21.814 | 1.700x | 4.193x |
| verify 32 | 52.335 | 29.359 | 1.783x | 4.806x |

Recomputed geometric mean across loads: 1.743000x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.6 µs  15.4%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.6 µs  14.4%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...      10.1 µs  11.5%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.6 µs   9.8%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.2 µs   5.9%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.2 µs   5.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   5.3%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.1 µs   4.7%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.1 µs   4.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   4.1%      1
_post_expert_norm_weighted_sum_kernel                              2.7 µs   3.1%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   2.8%      1
... 8 more
total                                                             88.0 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      23.4 µs  20.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      18.4 µs  16.0%      1
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.4 µs  11.6%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      12.3 µs  10.7%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.7 µs   8.4%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   4.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   3.9%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   3.4%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.9 µs   3.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              3.4 µs   2.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   2.1%      1
... 8 more
total                                                            115.1 µs
GPU busy (kernels overlap: PDL or streams)                       107.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      53.5 µs  31.9%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      30.8 µs  18.3%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      15.3 µs   9.1%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.8 µs   8.2%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.2 µs   6.1%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.8 µs   4.0%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.7 µs   3.4%      1
_matmul                                                            5.2 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.1%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.7%      1
... 7 more
total                                                            167.8 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.2 µs  39.5%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      41.7 µs  18.1%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      22.8 µs   9.9%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.5 µs   6.3%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.9 µs   6.0%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   4.4%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs   2.4%      1
_matmul                                                            5.2 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   1.8%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   1.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.5%      1
_ha__merge_norm_gate                                               3.1 µs   1.4%      1
... 7 more
total                                                            231.0 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_om__w2                                                          100.5 µs  51.4%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      18.0 µs   9.2%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.6 µs   8.0%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.9 µs   6.1%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.4 µs   4.8%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.3 µs   3.2%      1
_matmul                                                            5.4 µs   2.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.4%      1
_om__route                                                         2.6 µs   1.3%      1
... 6 more
total                                                            195.3 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     178.5 µs  48.8%      1
_om__w2                                                          104.0 µs  28.4%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.1 µs   4.7%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.5 µs   4.2%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.9 µs   3.2%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.0 µs   1.7%      1
_matmul                                                            5.2 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   1.1%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.8%      1
_om__route                                                         2.6 µs   0.7%      1
... 6 more
total                                                            366.0 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      46.8 µs  32.9%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      23.6 µs  16.6%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.5 µs   9.5%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.4 µs   6.6%      1
_ha__os__fused_attn                                                8.1 µs   5.7%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.8 µs   5.5%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   3.3%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   3.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              3.5 µs   2.5%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   1.8%      1
... 6 more
total                                                            142.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      55.3 µs  32.1%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      28.9 µs  16.8%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      18.0 µs  10.4%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.8 µs   8.0%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.5 µs   5.5%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.9 µs   4.6%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.5 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   2.8%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.1%      1
_ha__merge_norm_gate                                               2.9 µs   1.7%      1
... 8 more
total                                                            172.5 µs
GPU busy (kernels overlap: PDL or streams)                       168.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_om__w13_mix                                                     129.9 µs  41.5%      1
_om__w2_mix                                                       65.5 µs  20.9%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     23.1 µs   7.4%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.7 µs   6.3%      1
_ha__os__fused_attn                                               16.4 µs   5.2%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs   4.4%      1
_matmul                                                            7.5 µs   2.4%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.0 µs   2.2%      1
_om__route2                                                        6.2 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.2 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   1.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.4%      1
... 5 more
total                                                            313.2 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           152.9 µs  35.3%      1
_om__w13_mix                                                     117.4 µs  27.1%      1
_om__w2_mix                                                       63.2 µs  14.6%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.0 µs   5.1%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.7 µs   4.5%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.0 µs   3.2%      1
_matmul                                                            7.4 µs   1.7%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.6 µs   1.5%      1
_om__route2                                                        6.2 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.0 µs   1.2%      1
_post_expert_norm_weighted_sum_kernel                              4.7 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.1%      1
... 5 more
total                                                            432.8 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     181.1 µs  22.5%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     128.2 µs  15.9%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...      97.4 µs  12.1%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...      97.0 µs  12.0%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...      86.7 µs  10.8%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      67.8 µs   8.4%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           29.0 µs   3.6%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.7 µs   2.9%      1
_ha__os__fused_attn                                               21.3 µs   2.6%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.0 µs   2.5%      1
_matmul                                                           10.9 µs   1.4%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.9 µs   1.1%      1
... 9 more
total                                                            805.8 µs
GPU busy (kernels overlap: PDL or streams)                       433.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           293.1 µs  25.6%      1
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     175.0 µs  15.3%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     150.1 µs  13.1%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...     103.9 µs   9.1%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...     100.4 µs   8.8%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...      97.7 µs   8.5%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      97.0 µs   8.5%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.7 µs   2.5%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.7 µs   2.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.5 µs   1.8%      1
_matmul                                                           11.1 µs   1.0%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.6 µs   0.8%      1
... 9 more
total                                                           1144.0 µs
GPU busy (kernels overlap: PDL or streams)                       725.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

