# decoder_m256_wm_transfer_v1

Job: `6979bd1efba34f39`
Correctness: True; official speedup: 1.73597x; CI: [1.7099306485607937, 1.7620097546083304]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 172.98 | 89.36 | 1.936x | 40.02 |
| decode 1 | full | 181.66 | 96.61 | 1.880x | 44.18 |
| decode 16 | sliding | 265.06 | 157.04 | 1.688x | 76.72 |
| decode 16 | full | 433.18 | 236.14 | 1.834x | 137.46 |
| decode 32 | sliding | 268.26 | 195.68 | 1.371x | 84.51 |
| decode 32 | full | 606.75 | 356.01 | 1.704x | 208.10 |
| verify 1 | sliding | 248.11 | 129.99 | 1.909x | 62.29 |
| verify 1 | full | 271.25 | 147.13 | 1.844x | 66.44 |
| verify 16 | sliding | 522.64 | 297.08 | 1.759x | 126.61 |
| verify 16 | full | 709.81 | 433.93 | 1.636x | 188.68 |
| verify 32 | sliding | 720.74 | 408.14 | 1.766x | 143.81 |
| verify 32 | full | 1081.39 | 700.20 | 1.544x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.864 | 5.656 | 1.921x | 4.263x |
| decode 16 | 19.124 | 11.002 | 1.738x | 3.338x |
| decode 32 | 22.048 | 14.697 | 1.500x | 3.055x |
| verify 1 | 15.753 | 8.334 | 1.890x | 4.010x |
| verify 16 | 35.398 | 20.608 | 1.718x | 4.003x |
| verify 32 | 50.456 | 29.978 | 1.683x | 4.633x |

Recomputed geometric mean across loads: 1.735970x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      22.0 µs  21.2%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      14.9 µs  14.4%      1
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.6 µs  13.1%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.3 µs   9.0%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.1 µs   5.0%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   4.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   4.5%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.1 µs   4.0%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.7 µs   3.6%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   3.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.5%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.6%      1
... 8 more
total                                                            103.4 µs
GPU busy (kernels overlap: PDL or streams)                        98.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.7 µs  14.6%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      13.0 µs  13.9%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.9 µs  12.7%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.2 µs   9.8%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.4 µs   8.9%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   5.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   4.3%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.9 µs   4.2%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   4.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   3.8%      1
_post_expert_norm_weighted_sum_kernel                              3.1 µs   3.4%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.9 µs   3.1%      1
... 8 more
total                                                             93.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.2 µs  31.1%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.3 µs  16.9%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      15.3 µs   9.5%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.9 µs   8.6%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.2 µs   6.3%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.7 µs   4.2%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.7 µs   3.5%      1
_matmul                                                            5.2 µs   3.2%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.7%      1
... 7 more
total                                                            161.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      92.2 µs  38.9%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      45.4 µs  19.1%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      24.1 µs  10.2%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.6 µs   6.1%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.0 µs   5.9%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.0 µs   4.2%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.5 µs   2.3%      1
_matmul                                                            5.1 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.5%      1
_ha__merge_norm_gate                                               3.1 µs   1.3%      1
... 7 more
total                                                            237.0 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
_om__w2                                                           98.5 µs  51.2%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.6 µs   9.1%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.6 µs   8.1%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.7 µs   6.1%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.4 µs   4.9%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.2 µs   3.2%      1
_matmul                                                            5.6 µs   2.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.1%      1
_post_expert_norm_weighted_sum_kernel                              3.9 µs   2.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.4%      1
_ha__prepare_qkv                                                   2.4 µs   1.3%      1
... 6 more
total                                                            192.4 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     179.4 µs  49.2%      1
_om__w2                                                          102.1 µs  28.0%      2
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.1 µs   4.7%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.7 µs   4.3%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.7 µs   3.2%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.0 µs   1.7%      1
_matmul                                                            5.5 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.1%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.9 µs   0.8%      1
_ha__merge_norm_gate                                               2.4 µs   0.7%      1
... 6 more
total                                                            364.8 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      38.9 µs  29.1%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      23.3 µs  17.4%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.5 µs  10.1%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.1 µs   6.8%      1
_ha__os__fused_attn                                                8.3 µs   6.2%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.9 µs   5.9%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.1 µs   3.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   3.4%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   2.9%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   2.1%      1
... 6 more
total                                                            133.8 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      52.5 µs  31.6%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      27.3 µs  16.4%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.1 µs  10.3%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      15.0 µs   9.0%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.1 µs   5.5%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.7 µs   4.7%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   2.8%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   2.5%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   2.4%      1
_ha__merge_norm_gate                                               2.8 µs   1.7%      1
... 8 more
total                                                            166.3 µs
GPU busy (kernels overlap: PDL or streams)                       164.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_om__w13_mix                                                     103.5 µs  36.6%      1
_om__w2_mix                                                       61.6 µs  21.8%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.4 µs   7.9%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   6.9%      1
_ha__os__fused_attn                                               16.7 µs   5.9%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs   4.8%      1
_matmul                                                            7.3 µs   2.6%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.3 µs   2.6%      1
_om__route2                                                        6.3 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.2 µs   1.8%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   1.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   1.6%      1
... 5 more
total                                                            282.4 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           155.0 µs  33.8%      1
_om__w13_mix                                                     130.9 µs  28.6%      1
_om__w2_mix                                                       73.1 µs  16.0%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     21.4 µs   4.7%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           20.6 µs   4.5%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.7 µs   3.0%      1
_matmul                                                            7.2 µs   1.6%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.7 µs   1.5%      1
_om__route2                                                        6.3 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.0%      1
... 5 more
total                                                            458.2 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     166.8 µs  38.8%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     117.9 µs  27.4%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.6 µs   6.7%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.7 µs   5.5%      1
_ha__os__fused_attn                                               21.0 µs   4.9%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.9 µs   4.6%      1
_matmul                                                           10.8 µs   2.5%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       9.0 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.5 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.4 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.9 µs   0.7%      1
... 5 more
total                                                            430.1 µs
GPU busy (kernels overlap: PDL or streams)                       406.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           308.2 µs  40.1%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     192.1 µs  25.0%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     141.2 µs  18.4%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.6 µs   3.7%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.6 µs   3.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.8 µs   2.7%      1
_matmul                                                           11.3 µs   1.5%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       9.2 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.4 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.2 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              6.3 µs   0.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.4%      1
... 5 more
total                                                            768.8 µs
GPU busy (kernels overlap: PDL or streams)                       735.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

