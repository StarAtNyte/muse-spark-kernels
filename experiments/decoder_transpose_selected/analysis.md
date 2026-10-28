# decoder_transpose_selected

Job: `95f251028c7b4043`
Correctness: True; official speedup: 1.53450x; CI: [1.511484232312305, 1.5575192850730857]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 171.25 | 108.95 | 1.572x | 40.02 |
| decode 1 | full | 181.42 | 114.91 | 1.579x | 44.18 |
| decode 16 | sliding | 268.06 | 172.91 | 1.550x | 76.72 |
| decode 16 | full | 430.40 | 249.76 | 1.723x | 137.46 |
| decode 32 | sliding | 260.10 | 205.53 | 1.265x | 84.51 |
| decode 32 | full | 628.59 | 380.77 | 1.651x | 208.10 |
| verify 1 | sliding | 265.73 | 170.93 | 1.555x | 62.29 |
| verify 1 | full | 274.18 | 184.85 | 1.483x | 66.44 |
| verify 16 | sliding | 520.30 | 327.84 | 1.587x | 126.61 |
| verify 16 | full | 690.26 | 505.38 | 1.366x | 188.68 |
| verify 32 | sliding | 734.42 | 432.98 | 1.696x | 143.81 |
| verify 32 | full | 1068.22 | 778.61 | 1.372x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.780 | 6.850 | 1.574x | 4.231x |
| decode 16 | 19.217 | 11.950 | 1.608x | 3.355x |
| decode 32 | 22.022 | 15.547 | 1.416x | 3.051x |
| verify 1 | 16.610 | 10.820 | 1.535x | 4.228x |
| verify 16 | 34.978 | 23.167 | 1.510x | 3.955x |
| verify 32 | 50.875 | 32.375 | 1.571x | 4.672x |

Recomputed geometric mean across loads: 1.534502x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          43.4 µs  41.0%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.4 µs  11.8%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.6 µs   8.1%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.3 µs   6.0%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.1 µs   4.8%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.0 µs   4.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   4.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.5%      1
_post_expert_norm_weighted_sum_kernel                              2.9 µs   2.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   2.5%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.9 µs   1.8%      1
_merge_norm_gate                                                   1.8 µs   1.7%      1
... 6 more
total                                                            105.7 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          42.0 µs  37.7%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.6 µs  11.3%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      11.8 µs  10.6%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.3 µs   7.4%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.3 µs   5.6%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.1 µs   4.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   4.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.2%      1
_post_expert_norm_weighted_sum_kernel                              3.0 µs   2.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   2.4%      1
_merge_norm_gate                                                   2.3 µs   2.1%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.7 µs   1.5%      1
... 6 more
total                                                            111.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.4 µs  28.2%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.4 µs  15.3%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           25.6 µs  14.3%      2
_b5_matmul                                                        18.4 µs  10.3%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.3 µs   6.9%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.9 µs   3.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.6 µs   3.1%      1
_matmul                                                            4.8 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.5 µs   2.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   1.5%      1
... 8 more
total                                                            178.7 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      92.3 µs  34.8%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.9 µs  19.2%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.4 µs  10.3%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           25.4 µs   9.6%      2
_b5_matmul                                                        18.5 µs   7.0%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   4.7%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.5 µs   2.1%      1
_matmul                                                            4.9 µs   1.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.4%      1
_merge_norm_gate                                                   3.1 µs   1.2%      1
... 8 more
total                                                            265.1 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      52.7 µs  25.8%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      40.3 µs  19.7%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           27.1 µs  13.2%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.7 µs  10.1%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.7 µs   6.2%      1
void trans_decode<3>(unsigned char const*, unsigned char ...      10.0 µs   4.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.8 µs   2.8%      1
_matmul                                                            5.0 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   2.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       4.0 µs   2.0%      2
_post_expert_norm_weighted_sum_kernel                              3.7 µs   1.8%      1
... 7 more
total                                                            204.6 µs
GPU busy (kernels overlap: PDL or streams)                       201.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     178.3 µs  48.1%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      56.0 µs  15.1%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      35.3 µs   9.5%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           27.4 µs   7.4%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.8 µs   5.6%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.3 µs   3.3%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.0 µs   1.6%      1
_matmul                                                            5.3 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   1.1%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   1.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.9 µs   1.1%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.0%      1
... 7 more
total                                                            370.6 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      46.8 µs  25.7%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      23.5 µs  12.9%      1
_b5_matmul                                                        18.2 µs  10.0%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.7 µs   7.0%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.6 µs   6.9%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.5 µs   5.2%      1
_sliding_partials                                                  9.2 µs   5.1%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             6.8 µs   3.8%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      6.3 µs   3.5%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.0 µs   3.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.1%      1
... 11 more
total                                                            181.8 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      35.5 µs  19.2%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      21.2 µs  11.5%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      18.5 µs  10.0%      1
_b5_matmul                                                        18.4 µs   9.9%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.7 µs   6.9%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.5 µs   6.7%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.5 µs   5.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.3 µs   3.9%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.2 µs   3.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.7 µs   3.1%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.0 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   2.6%      1
... 13 more
total                                                            185.1 µs
GPU busy (kernels overlap: PDL or streams)                       179.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     127.5 µs  33.4%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     122.7 µs  32.1%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           28.3 µs   7.4%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.8 µs   6.5%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.2 µs   4.0%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.6 µs   3.8%      1
_matmul                                                            7.3 µs   1.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.8 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   1.2%      1
_post_expert_norm_weighted_sum_kernel                              4.5 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.0 µs   1.0%      1
... 9 more
total                                                            382.1 µs
GPU busy (kernels overlap: PDL or streams)                       337.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     192.2 µs  34.2%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     128.0 µs  22.8%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     122.8 µs  21.9%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           27.9 µs   5.0%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.7 µs   4.4%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.0 µs   2.7%      1
_matmul                                                            7.3 µs   1.3%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.7 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              5.5 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   0.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.5 µs   0.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   0.8%      1
... 9 more
total                                                            561.8 µs
GPU busy (kernels overlap: PDL or streams)                       517.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     190.0 µs  38.1%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     139.3 µs  28.0%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          33.7 µs   6.8%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           33.0 µs   6.6%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.6 µs   4.5%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.3 µs   4.1%      1
_matmul                                                           11.0 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.5 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.6 µs   1.5%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.1 µs   1.2%      1
_post_expert_norm_weighted_sum_kernel                              5.8 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs   0.9%      1
... 8 more
total                                                            498.1 µs
GPU busy (kernels overlap: PDL or streams)                       466.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     374.7 µs  45.6%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     154.9 µs  18.8%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     114.0 µs  13.9%      1
_matmul                                                           40.8 µs   5.0%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.0 µs   4.1%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           32.6 µs   4.0%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.3 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.3 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.9 µs   1.0%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.8 µs   0.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs   0.5%      1
_post_expert_norm_weighted_sum_kernel                              4.3 µs   0.5%      1
... 8 more
total                                                            822.0 µs
GPU busy (kernels overlap: PDL or streams)                       768.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

