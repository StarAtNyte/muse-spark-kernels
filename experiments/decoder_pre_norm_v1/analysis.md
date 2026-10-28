# decoder_pre_norm_v1

Job: `c0f2e3bd78904982`
Correctness: True; official speedup: 1.54683x; CI: [1.523630156522406, 1.570035135908875]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 170.30 | 103.37 | 1.648x | 40.02 |
| decode 1 | full | 180.08 | 110.29 | 1.633x | 44.18 |
| decode 16 | sliding | 263.65 | 168.18 | 1.568x | 76.72 |
| decode 16 | full | 441.42 | 258.06 | 1.711x | 137.46 |
| decode 32 | sliding | 261.89 | 203.75 | 1.285x | 84.51 |
| decode 32 | full | 618.38 | 373.82 | 1.654x | 208.10 |
| verify 1 | sliding | 258.82 | 162.11 | 1.597x | 62.29 |
| verify 1 | full | 278.90 | 182.79 | 1.526x | 66.44 |
| verify 16 | sliding | 507.55 | 327.49 | 1.550x | 126.61 |
| verify 16 | full | 680.78 | 502.80 | 1.354x | 188.68 |
| verify 32 | sliding | 700.72 | 420.69 | 1.666x | 143.81 |
| verify 32 | full | 1033.06 | 764.77 | 1.351x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.715 | 6.520 | 1.644x | 4.205x |
| decode 16 | 19.191 | 11.865 | 1.617x | 3.350x |
| decode 32 | 21.941 | 15.354 | 1.429x | 3.040x |
| verify 1 | 16.368 | 10.382 | 1.577x | 4.167x |
| verify 16 | 34.240 | 23.109 | 1.482x | 3.872x |
| verify 32 | 48.762 | 31.588 | 1.544x | 4.478x |

Recomputed geometric mean across loads: 1.546833x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          43.3 µs  43.2%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.4 µs  12.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       8.5 µs   8.5%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.1 µs   5.0%      1
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.9 µs   4.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   4.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   3.6%      1
_post_expert_norm_weighted_sum_kernel                              3.0 µs   3.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.5 µs   2.5%      1
_prepare_qkv                                                       2.0 µs   2.0%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.8 µs   1.8%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.7 µs   1.7%      1
... 6 more
total                                                            100.2 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
_b5_gemv                                                          42.3 µs  34.6%      4
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      20.0 µs  16.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      15.3 µs  12.5%      1
void fd2_partials<2>(unsigned char const*, unsigned char ...      12.4 µs  10.1%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.2 µs   4.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   3.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.7 µs   3.0%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.9%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   2.1%      1
_merge_norm_gate                                                   2.4 µs   2.0%      1
void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor...       1.7 µs   1.4%      1
_pre_rms_kernel                                                    1.6 µs   1.3%      1
... 6 more
total                                                            122.3 µs
GPU busy (kernels overlap: PDL or streams)                       117.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      43.5 µs  26.7%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           25.2 µs  15.5%      2
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      22.2 µs  13.6%      1
_b5_matmul                                                        19.4 µs  11.9%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.3 µs   7.5%      1
void trans_decode<3>(unsigned char const*, unsigned char ...       6.9 µs   4.3%      1
_matmul                                                            5.1 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.2%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.6 µs   1.6%      1
_prepare_qkv                                                       2.1 µs   1.3%      1
... 8 more
total                                                            163.2 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      91.3 µs  34.8%      1
void wm_gemm<0, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      50.5 µs  19.3%      1
void wm_gemm<1, 2, 2, 4, 0, 0>(uint4 const*, uint2 const*...      27.3 µs  10.4%      1
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           25.5 µs   9.7%      2
_b5_matmul                                                        19.4 µs   7.4%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    13.0 µs   5.0%      1
_matmul                                                            5.3 µs   2.0%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.4%      1
_merge_norm_gate                                                   3.1 µs   1.2%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   1.1%      1
... 8 more
total                                                            262.1 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      71.0 µs  32.2%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      42.8 µs  19.4%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           26.8 µs  12.2%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.3 µs   9.2%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.6 µs   5.7%      1
void trans_decode<3>(unsigned char const*, unsigned char ...      10.1 µs   4.6%      1
_matmul                                                            5.2 µs   2.3%      1
_post_expert_norm_weighted_sum_kernel                              4.8 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.9 µs   1.8%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.8%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.2%      1
... 7 more
total                                                            220.3 µs
GPU busy (kernels overlap: PDL or streams)                       214.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     180.9 µs  48.2%      1
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      59.1 µs  15.8%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      39.4 µs  10.5%      1
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           26.3 µs   7.0%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.3 µs   5.4%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   3.3%      1
_matmul                                                            5.3 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.0%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.9 µs   1.0%      2
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.8 µs   0.7%      1
... 7 more
total                                                            375.0 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      56.2 µs  27.9%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      39.6 µs  19.7%      1
_b5_matmul                                                        18.3 µs   9.1%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.7 µs   6.3%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.2 µs   6.1%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.4 µs   4.7%      1
_sliding_partials                                                  9.2 µs   4.6%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.1 µs   3.5%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.0 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.6 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.9%      1
_post_expert_norm_weighted_sum_kernel                              3.8 µs   1.9%      1
... 11 more
total                                                            201.7 µs
GPU busy (kernels overlap: PDL or streams)                       199.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      52.1 µs  25.7%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      26.5 µs  13.1%      1
_b5_matmul                                                        18.3 µs   9.0%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      18.2 µs   9.0%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      13.2 µs   6.5%      1
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    12.1 µs   6.0%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.3 µs   4.6%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.1 µs   3.5%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.1 µs   3.5%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.1 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.1 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.9%      1
... 13 more
total                                                            202.8 µs
GPU busy (kernels overlap: PDL or streams)                       199.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     123.6 µs  33.1%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     120.4 µs  32.2%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           27.8 µs   7.5%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           26.0 µs   7.0%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.2 µs   4.1%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.8 µs   4.0%      1
_matmul                                                            7.4 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.9 µs   1.3%      1
_post_expert_norm_weighted_sum_kernel                              4.8 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   1.3%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.2 µs   1.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.2 µs   0.8%      1
... 9 more
total                                                            373.6 µs
GPU busy (kernels overlap: PDL or streams)                       330.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     191.2 µs  34.5%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     125.4 µs  22.6%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     121.9 µs  22.0%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           27.6 µs   5.0%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           26.5 µs   4.8%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.1 µs   2.7%      1
_matmul                                                            7.6 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.7 µs   0.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.6 µs   0.8%      1
_post_expert_norm_weighted_sum_kernel                              4.4 µs   0.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   0.8%      1
void flash::prepare_varlen_num_blocks_kernel<1, true>(int...       3.2 µs   0.6%      1
... 9 more
total                                                            554.7 µs
GPU busy (kernels overlap: PDL or streams)                       506.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     172.8 µs  37.5%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     123.6 µs  26.8%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.1 µs   7.4%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           33.1 µs   7.2%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.4 µs   4.9%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.4 µs   4.4%      1
_matmul                                                           10.7 µs   2.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.4 µs   1.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.9 µs   1.7%      1
_post_expert_norm_weighted_sum_kernel                              5.2 µs   1.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.6 µs   1.0%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       3.2 µs   0.7%      1
... 8 more
total                                                            461.3 µs
GPU busy (kernels overlap: PDL or streams)                       431.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     373.6 µs  45.0%      1
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     169.2 µs  20.4%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     132.5 µs  16.0%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           35.2 µs   4.2%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.1 µs   4.1%      2
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.8 µs   2.7%      1
_matmul                                                           10.9 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.8 µs   1.1%      1
_finish                                                            8.1 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.6 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              5.5 µs   0.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs   0.5%      1
... 8 more
total                                                            830.4 µs
GPU busy (kernels overlap: PDL or streams)                       792.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

