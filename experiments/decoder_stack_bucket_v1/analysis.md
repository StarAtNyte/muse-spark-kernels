# decoder_stack_bucket_v1

Job: `a37d5a378c004866`
Correctness: True; official speedup: 1.72068x; CI: [1.69487312671924, 1.7464936280406382]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 171.01 | 87.94 | 1.945x | 40.02 |
| decode 1 | full | 181.02 | 96.09 | 1.884x | 44.18 |
| decode 16 | sliding | 266.13 | 158.85 | 1.675x | 76.72 |
| decode 16 | full | 442.72 | 245.05 | 1.807x | 137.46 |
| decode 32 | sliding | 264.02 | 194.66 | 1.356x | 84.51 |
| decode 32 | full | 619.68 | 332.90 | 1.861x | 208.10 |
| verify 1 | sliding | 259.86 | 138.76 | 1.873x | 62.29 |
| verify 1 | full | 274.16 | 150.04 | 1.827x | 66.44 |
| verify 16 | sliding | 530.16 | 320.15 | 1.656x | 126.61 |
| verify 16 | full | 693.90 | 454.97 | 1.525x | 188.68 |
| verify 32 | sliding | 717.38 | 407.87 | 1.759x | 143.81 |
| verify 32 | full | 1078.03 | 688.08 | 1.567x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.763 | 5.583 | 1.928x | 4.224x |
| decode 16 | 19.325 | 11.228 | 1.721x | 3.374x |
| decode 32 | 22.060 | 14.281 | 1.545x | 3.057x |
| verify 1 | 16.340 | 8.783 | 1.860x | 4.160x |
| verify 16 | 35.490 | 22.006 | 1.613x | 4.013x |
| verify 32 | 50.248 | 29.771 | 1.688x | 4.614x |

Recomputed geometric mean across loads: 1.720683x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      14.3 µs  16.2%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.6 µs  14.3%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.9 µs  11.2%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.6 µs   9.8%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.2 µs   5.8%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.9 µs   5.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   5.2%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.1 µs   4.6%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       3.9 µs   4.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   4.2%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   3.6%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   3.0%      1
... 8 more
total                                                             88.6 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.6 µs  14.4%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.8 µs  13.5%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.9 µs  12.6%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.8 µs  10.4%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.6 µs   9.1%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   5.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   4.9%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.0 µs   4.2%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.0 µs   4.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.8%      1
_post_expert_norm_weighted_sum_kernel                              3.1 µs   3.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.8%      1
... 8 more
total                                                             94.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.9 µs  31.6%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      28.0 µs  17.4%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      15.1 µs   9.4%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.9 µs   8.6%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...       9.9 µs   6.1%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.9 µs   4.3%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs   3.5%      1
_matmul                                                            5.1 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   2.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   1.6%      1
... 7 more
total                                                            161.0 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                            96.5 µs  39.1%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      51.5 µs  20.9%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.5 µs  11.2%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.1 µs   5.7%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.0 µs   5.7%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   4.1%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.7 µs   2.3%      1
_matmul                                                            5.1 µs   2.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              4.1 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.4%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   1.1%      1
... 5 more
total                                                            246.7 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      59.8 µs  29.7%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      47.3 µs  23.5%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.9 µs   8.9%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.5 µs   7.7%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.6 µs   5.8%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.7 µs   4.8%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.4 µs   3.2%      1
_matmul                                                            5.3 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   2.3%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.4%      1
... 7 more
total                                                            201.4 µs
GPU busy (kernels overlap: PDL or streams)                       193.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                           184.6 µs  49.8%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      59.9 µs  16.2%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      47.1 µs  12.7%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      16.6 µs   4.5%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.4 µs   4.2%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      12.1 µs   3.3%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       5.9 µs   1.6%      1
_matmul                                                            5.4 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.8%      1
... 5 more
total                                                            370.8 µs
GPU busy (kernels overlap: PDL or streams)                       362.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      47.2 µs  31.9%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      24.5 µs  16.5%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.4 µs   9.1%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.2 µs   6.2%      1
_ha__sliding_partials                                              8.8 µs   6.0%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.7 µs   5.2%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.3 µs   3.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.9 µs   3.3%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   2.9%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   2.5%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   1.7%      1
... 8 more
total                                                            148.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      37.8 µs  25.2%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      27.0 µs  17.9%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.4 µs  11.6%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      14.1 µs   9.4%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.2 µs   6.1%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.7 µs   5.1%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.3 µs   3.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   3.1%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.3 µs   2.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              3.0 µs   2.0%      1
_ha__merge_norm_gate                                               2.8 µs   1.9%      1
... 8 more
total                                                            150.3 µs
GPU busy (kernels overlap: PDL or streams)                       145.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     131.7 µs  34.7%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     130.6 µs  34.4%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     23.7 µs   6.3%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   5.2%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      16.9 µs   4.4%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.6 µs   3.6%      1
_matmul                                                            7.4 µs   1.9%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.3 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.8 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              4.7 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   1.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.8%      1
... 6 more
total                                                            379.3 µs
GPU busy (kernels overlap: PDL or streams)                       340.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           155.1 µs  30.9%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     127.2 µs  25.4%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     123.3 µs  24.6%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.5 µs   4.5%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   3.9%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.5 µs   2.9%      1
_matmul                                                            7.4 µs   1.5%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.8 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              4.7 µs   0.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   0.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.6%      1
... 5 more
total                                                            501.7 µs
GPU busy (kernels overlap: PDL or streams)                       456.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     139.9 µs  17.5%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...     122.2 µs  15.3%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     119.9 µs  15.0%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...     111.7 µs  14.0%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      92.1 µs  11.5%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...      55.4 µs   6.9%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           29.5 µs   3.7%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.4 µs   2.9%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.2 µs   2.5%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.2 µs   2.5%      1
_matmul                                                           10.9 µs   1.4%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.7 µs   1.1%      1
... 13 more
total                                                            798.3 µs
GPU busy (kernels overlap: PDL or streams)                       475.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           305.1 µs  27.9%      1
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     143.4 µs  13.1%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     131.7 µs  12.1%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...     128.7 µs  11.8%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...     126.3 µs  11.6%      1
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...      85.6 µs   7.8%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...      45.3 µs   4.1%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.0 µs   2.6%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.5 µs   2.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.5 µs   1.9%      1
_matmul                                                           11.9 µs   1.1%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.6 µs   0.8%      1
... 9 more
total                                                           1092.2 µs
GPU busy (kernels overlap: PDL or streams)                       769.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

