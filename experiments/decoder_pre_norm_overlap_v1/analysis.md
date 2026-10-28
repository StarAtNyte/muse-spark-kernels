# decoder_pre_norm_overlap_v1

Job: `ed021d8da1d045bf`
Correctness: True; official speedup: 1.56755x; CI: [1.5440397555677534, 1.5910663471078879]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 169.26 | 100.30 | 1.688x | 40.02 |
| decode 1 | full | 180.91 | 110.32 | 1.640x | 44.18 |
| decode 16 | sliding | 257.31 | 162.23 | 1.586x | 76.72 |
| decode 16 | full | 438.72 | 254.07 | 1.727x | 137.46 |
| decode 32 | sliding | 258.91 | 200.11 | 1.294x | 84.51 |
| decode 32 | full | 610.16 | 366.31 | 1.666x | 208.10 |
| verify 1 | sliding | 264.93 | 166.65 | 1.590x | 62.29 |
| verify 1 | full | 276.66 | 180.54 | 1.532x | 66.44 |
| verify 16 | sliding | 519.12 | 327.06 | 1.587x | 126.61 |
| verify 16 | full | 697.01 | 502.09 | 1.388x | 188.68 |
| verify 32 | sliding | 711.50 | 417.43 | 1.704x | 143.81 |
| verify 32 | full | 1077.76 | 781.51 | 1.379x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.681 | 6.379 | 1.674x | 4.192x |
| decode 16 | 18.856 | 11.528 | 1.636x | 3.292x |
| decode 32 | 21.673 | 15.066 | 1.439x | 3.003x |
| verify 1 | 16.613 | 10.554 | 1.574x | 4.229x |
| verify 16 | 35.032 | 23.078 | 1.518x | 3.961x |
| verify 32 | 49.973 | 31.706 | 1.576x | 4.589x |

Recomputed geometric mean across loads: 1.567553x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          43.3 µs  42.8%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.8 µs  12.7%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.5 µs   8.4%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.0 µs   5.0%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       5.0 µs   4.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   4.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.4 µs   3.4%      1
_post_expert_norm_weighted_sum_kernel                              3.1 µs   3.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   2.6%      1
_prepare_qkv                                                       2.0 µs   2.0%      1
_pre_rms_kernel                                                    2.0 µs   2.0%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.9 µs   1.9%      1
... 6 more
total                                                            101.2 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          42.8 µs  34.9%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      19.9 µs  16.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      14.9 µs  12.2%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      12.0 µs   9.8%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.2 µs   4.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.5 µs   3.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.0%      1
_post_expert_norm_weighted_sum_kernel                              3.5 µs   2.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   2.1%      1
_merge_norm_gate                                                   2.5 µs   2.1%      1
_pre_rms_kernel                                                    1.8 µs   1.5%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.8 µs   1.4%      1
... 6 more
total                                                            122.4 µs
GPU busy (kernels overlap: PDL or streams)                       117.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      45.9 µs  27.3%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           26.4 µs  15.7%      2
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      24.0 µs  14.2%      1
_b5_matmul                                                        19.0 µs  11.3%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.3 µs   7.3%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.9 µs   4.1%      1
_matmul                                                            4.9 µs   2.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   2.4%      1
_post_expert_norm_weighted_sum_kernel                              3.7 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   2.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.7%      1
_prepare_qkv                                                       2.1 µs   1.3%      1
... 8 more
total                                                            168.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.4 µs  34.7%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.9 µs  19.3%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.2 µs  10.3%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           26.0 µs   9.9%      2
_b5_matmul                                                        19.7 µs   7.5%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   4.7%      1
_post_expert_norm_weighted_sum_kernel                              4.9 µs   1.8%      1
_matmul                                                            4.8 µs   1.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.6%      1
_merge_norm_gate                                                   3.8 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.0%      1
... 8 more
total                                                            263.6 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      74.4 µs  32.8%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      44.7 µs  19.7%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           26.9 µs  11.9%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.6 µs   9.1%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.5 µs   5.5%      1
void trans_decode<3>(unsigned char const*, unsigned char ...      10.1 µs   4.5%      1
_matmul                                                            5.2 µs   2.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       4.8 µs   2.1%      2
_post_expert_norm_weighted_sum_kernel                              4.3 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.7%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.3%      1
... 7 more
total                                                            226.6 µs
GPU busy (kernels overlap: PDL or streams)                       222.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     181.0 µs  48.9%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      52.0 µs  14.1%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      40.0 µs  10.8%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           26.9 µs   7.3%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   21.1 µs   5.7%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   3.4%      1
_matmul                                                            5.4 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.9 µs   1.1%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.8%      1
... 7 more
total                                                            369.9 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      38.9 µs  22.6%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      26.8 µs  15.6%      1
_b5_matmul                                                        18.6 µs  10.8%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.5 µs   7.3%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.5 µs   7.3%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.4 µs   5.5%      1
_sliding_partials                                                  9.3 µs   5.4%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.1 µs   4.1%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.1 µs   3.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   2.2%      1
... 11 more
total                                                            172.4 µs
GPU busy (kernels overlap: PDL or streams)                       167.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      55.9 µs  26.0%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      32.9 µs  15.3%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      18.9 µs   8.8%      1
_b5_matmul                                                        18.4 µs   8.5%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.7 µs   5.9%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.6 µs   5.9%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.5 µs   4.4%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.5 µs   3.5%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.0 µs   3.3%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.1 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.0%      1
... 13 more
total                                                            214.9 µs
GPU busy (kernels overlap: PDL or streams)                       207.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     124.9 µs  32.8%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     124.8 µs  32.8%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           28.8 µs   7.6%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.3 µs   6.7%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    16.2 µs   4.3%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.6 µs   3.8%      1
_matmul                                                            7.4 µs   1.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.7 µs   1.2%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   1.2%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.1 µs   0.8%      1
... 9 more
total                                                            380.2 µs
GPU busy (kernels overlap: PDL or streams)                       335.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     195.3 µs  34.9%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     128.9 µs  23.0%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     121.7 µs  21.7%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           28.4 µs   5.1%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           24.6 µs   4.4%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.5 µs   2.8%      1
_matmul                                                            7.3 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   0.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   0.8%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   0.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.9 µs   0.7%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.2 µs   0.6%      1
... 9 more
total                                                            560.1 µs
GPU busy (kernels overlap: PDL or streams)                       518.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     159.5 µs  37.3%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     105.5 µs  24.7%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.4 µs   8.0%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           31.1 µs   7.3%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           23.4 µs   5.5%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      19.7 µs   4.6%      1
_matmul                                                           10.9 µs   2.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.5 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.4 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.2%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.0 µs   0.7%      1
... 8 more
total                                                            427.8 µs
GPU busy (kernels overlap: PDL or streams)                       404.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     377.4 µs  43.6%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     180.9 µs  20.9%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     137.7 µs  15.9%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.1 µs   3.9%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           33.1 µs   3.8%      1
_matmul                                                           30.2 µs   3.5%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           23.0 µs   2.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       9.0 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.6 µs   1.0%      1
_post_expert_norm_weighted_sum_kernel                              6.1 µs   0.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.2 µs   0.5%      1
_pre_rms_kernel                                                    4.1 µs   0.5%      1
... 8 more
total                                                            865.7 µs
GPU busy (kernels overlap: PDL or streams)                       809.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

