# decoder_attention_split4_b32_v1

Job: `3e5c2ae3875945c1`
Correctness: True; official speedup: 1.69848x; CI: [1.6730046829165213, 1.7239591402642322]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 173.60 | 89.40 | 1.942x | 40.02 |
| decode 1 | full | 184.80 | 98.74 | 1.872x | 44.18 |
| decode 16 | sliding | 263.38 | 158.05 | 1.666x | 76.72 |
| decode 16 | full | 436.32 | 242.22 | 1.801x | 137.46 |
| decode 32 | sliding | 255.17 | 192.61 | 1.325x | 84.51 |
| decode 32 | full | 614.88 | 364.44 | 1.687x | 208.10 |
| verify 1 | sliding | 269.50 | 144.96 | 1.859x | 62.29 |
| verify 1 | full | 280.03 | 154.23 | 1.816x | 66.44 |
| verify 16 | sliding | 526.85 | 316.90 | 1.662x | 126.61 |
| verify 16 | full | 694.19 | 450.25 | 1.542x | 188.68 |
| verify 32 | sliding | 721.95 | 425.31 | 1.697x | 143.81 |
| verify 32 | full | 1084.50 | 685.83 | 1.581x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.942 | 5.692 | 1.922x | 4.294x |
| decode 16 | 19.096 | 11.146 | 1.713x | 3.334x |
| decode 32 | 21.576 | 14.691 | 1.469x | 2.990x |
| verify 1 | 16.878 | 9.136 | 1.847x | 4.296x |
| verify 16 | 35.342 | 21.782 | 1.623x | 3.997x |
| verify 32 | 50.562 | 30.538 | 1.656x | 4.643x |

Recomputed geometric mean across loads: 1.698482x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.6 µs  15.5%      1
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.6 µs  14.5%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.6 µs  11.0%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.6 µs   9.9%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   5.8%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.9 µs   5.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   5.2%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.0 µs   4.6%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       3.9 µs   4.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   4.1%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   3.6%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   3.1%      1
... 8 more
total                                                             87.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      22.2 µs  19.6%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      16.9 µs  14.9%      1
void v18s_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, ...      13.5 µs  12.0%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.9 µs  10.5%      1
void v18d_native_gemv<1, 1, 1, 2, 2, 0, 256, 0, 2304, 819...       9.7 µs   8.5%      1
void smallm2_gemv<1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, ...       5.1 µs   4.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   4.1%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1...       4.2 µs   3.7%      1
void fp32_router<1, 512, 2>(float const*, float const*, f...       4.1 µs   3.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   3.2%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   3.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.3%      1
... 8 more
total                                                            113.2 µs
GPU busy (kernels overlap: PDL or streams)                       106.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      55.4 µs  32.7%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      30.8 µs  18.1%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      16.0 µs   9.4%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      13.8 µs   8.1%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.1 µs   5.9%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.6 µs   3.9%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       5.6 µs   3.3%      1
_matmul                                                            5.3 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.1%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.6%      1
... 7 more
total                                                            169.6 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                            96.0 µs  41.3%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      42.2 µs  18.2%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      22.6 µs   9.7%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 4096,...      14.4 µs   6.2%      1
void dd_cap2_b_cpssm_gemv<1, 2, 1, 2, 0, 0, 256, 0, 8192,...      14.2 µs   6.1%      1
void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<1, 2, 1...      10.0 µs   4.3%      1
void m16trace_59fcb5b72994_gemv<1, 2, 1, 1, 0, 0, 256, 0,...       6.0 µs   2.6%      1
_matmul                                                            5.2 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              4.3 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   1.5%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.2%      1
... 5 more
total                                                            232.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      51.7 µs  27.3%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      43.3 µs  22.9%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      18.0 µs   9.5%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.8 µs   8.3%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.9 µs   6.3%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       9.9 µs   5.2%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.3 µs   3.3%      1
_matmul                                                            5.6 µs   2.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              3.9 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.4%      1
... 7 more
total                                                            189.6 µs
GPU busy (kernels overlap: PDL or streams)                       186.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_decode                                           185.0 µs  52.4%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      55.4 µs  15.7%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      33.9 µs   9.6%      1
void r3_pool32_pooler_d9_cap2_a_m32x_gemv<1, 4, 1, 1, 0, ...      17.2 µs   4.9%      1
void m32w_gemm<32, 64, 4, 2, 2, true, 1>(CUtensorMap_st, ...      15.6 µs   4.4%      1
void m32x_q_gemv<1, 4, 1, 1, 5, 0, 128, 0, 2304, 8192, 1,...      11.9 µs   3.4%      1
void r3_o32_o_direct_a_ol_projection<32, 128, 8, 1, 1, 1,...       6.1 µs   1.7%      1
_matmul                                                            5.4 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   1.2%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.1%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   0.8%      1
... 5 more
total                                                            353.3 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      37.8 µs  27.4%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      22.8 µs  16.6%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.6 µs   9.8%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.4 µs   6.8%      1
_ha__sliding_partials                                              8.8 µs   6.4%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       8.5 µs   6.2%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.2 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   3.3%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.8%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   2.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.0%      1
... 8 more
total                                                            138.0 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      48.0 µs  30.3%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      23.8 µs  15.0%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      17.3 µs  10.9%      1
void smallm2_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, ...      13.9 µs   8.8%      1
void qs_literal_owner5_decode5_gemv<1, 1, 1, 2, 5, 0, 256...       9.4 µs   5.9%      1
void fp32_router<8, 256, 1>(float const*, float const*, f...       7.7 µs   4.9%      1
void v18s_gemv<1, 1, 1, 2, 0, 0, 256, 0, 8192, 1024, 1, 8...       5.8 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   3.0%      1
void smtrace_gemv<1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1...       4.2 µs   2.6%      1
_post_expert_norm_weighted_sum_kernel                              3.8 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.4%      1
_ha__merge_norm_gate                                               2.8 µs   1.8%      1
... 8 more
total                                                            158.4 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     133.6 µs  36.3%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     118.3 µs  32.1%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.9 µs   6.2%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.6 µs   5.3%      1
wg5_partials(unsigned char const*, unsigned char const*, ...      16.9 µs   4.6%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.8 µs   3.7%      1
_matmul                                                            7.3 µs   2.0%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.0 µs   1.9%      1
_post_expert_norm_weighted_sum_kernel                              5.4 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.4 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   1.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.1 µs   0.8%      1
... 6 more
total                                                            367.9 µs
GPU busy (kernels overlap: PDL or streams)                       328.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           153.4 µs  30.5%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     127.6 µs  25.4%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     126.9 µs  25.2%      1
nvjet_sm90_tst_64x64_64x13_1x1_v_bz_coopB_NNN                     22.0 µs   4.4%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           19.7 µs   3.9%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      13.8 µs   2.7%      1
_matmul                                                            7.5 µs   1.5%      1
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       6.6 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.5 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   0.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.6%      1
... 5 more
total                                                            503.0 µs
GPU busy (kernels overlap: PDL or streams)                       468.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     165.1 µs  37.6%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     117.4 µs  26.7%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           28.8 µs   6.6%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.7 µs   5.4%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.2 µs   4.6%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.1 µs   4.6%      1
_matmul                                                           10.9 µs   2.5%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.7 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.4 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.6 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              5.2 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.6 µs   1.0%      1
... 9 more
total                                                            439.1 µs
GPU busy (kernels overlap: PDL or streams)                       413.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_ha__os__g_full_verify                                           296.2 µs  40.6%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     177.4 µs  24.3%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     131.5 µs  18.0%      1
nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NNN                           27.8 µs   3.8%      1
nvjet_sm90_tst_128x128_64x6_1x1_v_bz_coopA_NNN                    23.6 µs   3.2%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.3 µs   2.8%      1
_matmul                                                           11.1 µs   1.5%      1
void rl_early_projection<256, 64, 5, 7, 1, 1, 1>(CUtensor...       8.6 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.4 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.1 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              6.3 µs   0.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.4%      1
... 5 more
total                                                            729.9 µs
GPU busy (kernels overlap: PDL or streams)                       702.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

