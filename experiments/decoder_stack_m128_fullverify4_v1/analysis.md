# decoder_stack_m128_fullverify4_v1

Job: `4afc1ae8979649aa`
Correctness: True; official speedup: 1.73954x; CI: [1.7134458135960178, 1.7656319805075713]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 172.02 | 89.27 | 1.927x | 40.02 |
| decode 1 | full | 181.68 | 97.41 | 1.865x | 44.18 |
| decode 16 | sliding | 263.65 | 157.04 | 1.679x | 76.72 |
| decode 16 | full | 438.72 | 240.03 | 1.828x | 137.46 |
| decode 32 | sliding | 267.36 | 194.73 | 1.373x | 84.51 |
| decode 32 | full | 622.75 | 344.99 | 1.805x | 208.10 |
| verify 1 | sliding | 263.22 | 136.04 | 1.935x | 62.29 |
| verify 1 | full | 278.58 | 153.79 | 1.811x | 66.44 |
| verify 16 | sliding | 515.42 | 304.61 | 1.692x | 126.61 |
| verify 16 | full | 720.64 | 447.32 | 1.611x | 188.68 |
| verify 32 | sliding | 694.14 | 385.06 | 1.803x | 143.81 |
| verify 32 | full | 1061.63 | 666.71 | 1.592x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.820 | 5.665 | 1.910x | 4.246x |
| decode 16 | 19.147 | 11.064 | 1.731x | 3.342x |
| decode 32 | 22.263 | 14.477 | 1.538x | 3.085x |
| verify 1 | 16.565 | 8.719 | 1.900x | 4.217x |
| verify 16 | 35.240 | 21.169 | 1.665x | 3.985x |
| verify 32 | 48.917 | 28.380 | 1.724x | 4.492x |

Recomputed geometric mean across loads: 1.739539x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.4 µs  15.6%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.5 µs  14.6%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.6 µs  11.2%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.2 µs   9.5%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   5.9%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.9 µs   5.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   5.0%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.0 µs   4.7%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.8 µs   4.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   4.1%      1
_post_expert_norm_weighted_sum_kernel                              3.1 µs   3.6%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   2.9%      1
... 8 more
total                                                             85.9 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.5 µs  14.3%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.5 µs  13.3%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.9 µs  12.7%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.6 µs  10.2%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.3 µs   8.8%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.0 µs   5.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   5.0%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.9 µs   4.1%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   4.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   3.9%      1
_post_expert_norm_weighted_sum_kernel                              3.1 µs   3.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   2.7%      1
... 8 more
total                                                             94.2 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      49.6 µs  30.7%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      28.3 µs  17.5%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.8 µs   9.2%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.8 µs   8.6%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.2 µs   6.3%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.7 µs   4.2%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.4 µs   3.4%      1
_matmul                                                            5.2 µs   3.2%      1
_post_expert_norm_weighted_sum_kernel                              4.9 µs   3.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.6%      1
... 7 more
total                                                            161.2 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      89.3 µs  38.5%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      41.9 µs  18.1%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      23.2 µs  10.0%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      15.1 µs   6.5%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.9 µs   6.4%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.0 µs   4.3%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.5 µs   2.4%      1
_matmul                                                            5.1 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.6%      1
_ha__merge_norm_gate                                               2.8 µs   1.2%      1
... 7 more
total                                                            231.8 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_om__w2                                                          107.8 µs  52.9%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.5 µs   8.6%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.6 µs   7.6%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      12.6 µs   6.2%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.5 µs   4.7%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       7.1 µs   3.5%      1
_matmul                                                            5.5 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   2.1%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.4%      1
_om__route                                                         2.6 µs   1.3%      1
... 6 more
total                                                            203.8 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     176.3 µs  47.1%      1
_om__w2                                                          114.7 µs  30.6%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.8 µs   4.5%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.5 µs   4.1%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.8 µs   3.1%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.0 µs   1.6%      1
_matmul                                                            5.4 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.7%      1
_ha__merge_norm_gate                                               2.6 µs   0.7%      1
... 6 more
total                                                            374.4 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      38.3 µs  29.0%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      22.0 µs  16.6%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.4 µs  10.2%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.3 µs   7.1%      1
_ha__os__fused_attn                                                8.3 µs   6.3%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.9 µs   6.0%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.0 µs   3.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   3.6%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   2.8%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.9 µs   2.2%      1
... 6 more
total                                                            132.0 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      48.3 µs  30.7%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      23.8 µs  15.1%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.2 µs  10.9%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.9 µs   8.8%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.3 µs   5.9%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.8 µs   5.0%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.0 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   3.0%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.1 µs   2.6%      1
_post_expert_norm_weighted_sum_kernel                              3.8 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.4%      1
_ha__merge_norm_gate                                               2.8 µs   1.8%      1
... 8 more
total                                                            157.5 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_om__w13_mix                                                     120.7 µs  40.3%      1
_om__w2_mix                                                       62.2 µs  20.8%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.7 µs   7.6%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.4 µs   6.5%      1
_ha__os__fused_attn                                               16.7 µs   5.6%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs   4.5%      1
_matmul                                                            7.4 µs   2.5%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.0 µs   2.3%      1
_om__route2                                                        6.4 µs   2.1%      1
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.5%      1
... 5 more
total                                                            299.6 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           154.2 µs  33.9%      1
_om__w13_mix                                                     130.5 µs  28.7%      1
_om__w2_mix                                                       71.5 µs  15.7%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     21.3 µs   4.7%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.3 µs   4.2%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs   3.0%      1
_matmul                                                            7.4 µs   1.6%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.9 µs   1.5%      1
_om__route2                                                        6.4 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              5.4 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.0%      1
... 5 more
total                                                            455.0 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     180.7 µs  22.2%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     124.5 µs  15.3%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...      99.4 µs  12.2%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...      97.0 µs  11.9%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...      87.1 µs  10.7%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      77.8 µs   9.6%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           29.0 µs   3.6%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.8 µs   2.9%      1
_ha__os__fused_attn                                               21.1 µs   2.6%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.9 µs   2.4%      1
_matmul                                                           10.8 µs   1.3%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.7 µs   1.1%      1
... 9 more
total                                                            813.8 µs
GPU busy (kernels overlap: PDL or streams)                       431.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           293.0 µs  26.5%      1
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     166.9 µs  15.1%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     141.4 µs  12.8%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...     101.1 µs   9.1%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...      99.1 µs   8.9%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...      96.3 µs   8.7%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      84.1 µs   7.6%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           27.7 µs   2.5%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.5 µs   2.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.2 µs   1.8%      1
_matmul                                                           11.1 µs   1.0%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       9.0 µs   0.8%      1
... 9 more
total                                                           1107.3 µs
GPU busy (kernels overlap: PDL or streams)                       721.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

