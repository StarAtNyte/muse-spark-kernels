# decode_attention_hybrid_v4

Job: `0dd0d5d2f7aa4430`
Correctness: True; official speedup: 1.15395x; CI: [1.1366414911983738, 1.1712600137729434]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 170.42 | 136.14 | 1.252x | 40.02 |
| decode 1 | full | 180.64 | 148.85 | 1.214x | 44.18 |
| decode 16 | sliding | 260.10 | 215.94 | 1.205x | 76.72 |
| decode 16 | full | 438.05 | 383.39 | 1.143x | 137.46 |
| decode 32 | sliding | 263.04 | 249.61 | 1.054x | 84.51 |
| decode 32 | full | 619.68 | 583.18 | 1.063x | 208.10 |
| verify 1 | sliding | 266.24 | 229.41 | 1.161x | 62.29 |
| verify 1 | full | 272.18 | 238.62 | 1.141x | 66.44 |
| verify 16 | sliding | 504.42 | 417.71 | 1.208x | 126.61 |
| verify 16 | full | 695.86 | 612.13 | 1.137x | 188.68 |
| verify 32 | sliding | 694.93 | 614.14 | 1.132x | 143.81 |
| verify 32 | full | 1050.00 | 970.99 | 1.081x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.729 | 8.644 | 1.241x | 4.211x |
| decode 16 | 18.973 | 16.067 | 1.181x | 3.312x |
| decode 32 | 22.015 | 20.813 | 1.058x | 3.050x |
| verify 1 | 16.602 | 14.371 | 1.155x | 4.226x |
| verify 16 | 34.337 | 29.009 | 1.184x | 3.883x |
| verify 32 | 48.767 | 43.786 | 1.114x | 4.478x |

Recomputed geometric mean across loads: 1.153951x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      44.9 µs  31.8%      2
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    32.2 µs  22.8%      2
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             7.0 µs   4.9%      1
_sliding_partials                                                  6.1 µs   4.4%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.0 µs   4.3%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.2 µs   3.7%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.0 µs   3.6%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       5.0 µs   3.5%      3
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   3.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   2.6%      1
_post_expert_norm_weighted_sum_kernel                              3.4 µs   2.4%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.9%      1
... 11 more
total                                                            141.1 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      45.2 µs  29.4%      2
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    32.3 µs  21.0%      2
_sliding_partials                                                 17.7 µs  11.5%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             6.8 µs   4.4%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.8 µs   3.8%      1
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.2 µs   3.4%      1
void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv...       5.1 µs   3.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       5.0 µs   3.3%      3
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.9 µs   3.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.0 µs   2.6%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.3%      1
void sglang::muse_spark::routing_kernel<sglang::muse_spar...       2.7 µs   1.8%      1
... 11 more
total                                                            153.5 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     107.6 µs  46.4%      2
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           26.2 µs  11.3%      2
nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT                   20.1 µs   8.6%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      13.1 µs   5.6%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   5.3%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.3 µs   2.7%      1
_matmul                                                            5.7 µs   2.5%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.8%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.1 µs   1.8%      1
_post_expert_norm_weighted_sum_kernel                              3.8 µs   1.6%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.6 µs   1.6%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   1.5%      1
... 12 more
total                                                            232.2 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                182.4 µs  46.1%      1
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     108.3 µs  27.4%      2
nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT                           27.5 µs   6.9%      2
nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT                   20.0 µs   5.1%      1
nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT                    12.2 µs   3.1%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.6 µs   1.4%      1
_matmul                                                            5.0 µs   1.3%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.2 µs   1.1%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       3.6 µs   0.9%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              3.3 µs   0.8%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.7%      1
... 10 more
total                                                            395.8 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     124.2 µs  48.4%      2
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           26.8 µs  10.5%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   20.5 µs   8.0%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      18.9 µs   7.4%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.5 µs   4.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.1 µs   2.4%      1
_matmul                                                            5.2 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   1.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       4.0 µs   1.6%      2
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.8 µs   1.5%      1
_post_expert_norm_weighted_sum_kernel                              3.6 µs   1.4%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       3.0 µs   1.2%      1
... 12 more
total                                                            256.4 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                353.4 µs  58.0%      1
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     135.2 µs  22.2%      2
nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT                           27.6 µs   4.5%      2
nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT                   21.6 µs   3.5%      1
nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT                    12.4 µs   2.0%      1
_finish                                                           11.1 µs   1.8%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.9 µs   1.0%      1
_matmul                                                            5.3 µs   0.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.3 µs   0.7%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       4.0 µs   0.7%      2
_post_expert_norm_weighted_sum_kernel                              3.7 µs   0.6%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.6 µs   0.6%      1
... 10 more
total                                                            609.0 µs
GPU busy (kernels overlap: PDL or streams)                       596.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      86.7 µs  41.9%      2
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    32.6 µs  15.8%      2
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      12.6 µs   6.1%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.6 µs   4.6%      1
_sliding_partials                                                  9.3 µs   4.5%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             6.9 µs   3.4%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.7 µs   2.8%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       5.2 µs   2.5%      3
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.0 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.9 µs   2.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.1 µs   2.0%      1
_moe_align_small_numel_kernel                                      3.5 µs   1.7%      1
... 11 more
total                                                            206.8 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      96.9 µs  40.9%      2
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT                    33.0 µs  14.0%      2
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      18.8 µs   7.9%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...      13.2 µs   5.6%      1
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta...       9.6 µs   4.1%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       7.6 µs   3.2%      1
nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT                             6.9 µs   2.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.5 µs   2.3%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, __...       5.3 µs   2.3%      3
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT                      5.1 µs   2.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.8 µs   2.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       3.9 µs   1.6%      1
... 13 more
total                                                            236.6 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     295.4 µs  66.8%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           29.2 µs   6.6%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.0 µs   5.6%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.0 µs   3.4%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      14.2 µs   3.2%      1
_matmul                                                            7.3 µs   1.7%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.2 µs   1.4%      1
_post_expert_norm_weighted_sum_kernel                              5.2 µs   1.2%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.0 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   1.0%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.2 µs   1.0%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       4.1 µs   0.9%      1
... 12 more
total                                                            442.1 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     283.9 µs  46.7%      2
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     192.4 µs  31.6%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           28.0 µs   4.6%      2
nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT                           25.0 µs   4.1%      1
nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN                    15.1 µs   2.5%      1
_matmul                                                            7.1 µs   1.2%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.1 µs   1.0%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       5.4 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              5.3 µs   0.9%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 256, ...       4.4 µs   0.7%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.3 µs   0.7%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.7 µs   0.6%      1
... 12 more
total                                                            607.8 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     455.5 µs  71.4%      2
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          34.0 µs   5.3%      2
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           31.8 µs   5.0%      1
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.5 µs   3.5%      1
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.5 µs   3.2%      1
_matmul                                                           10.8 µs   1.7%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.7 µs   1.4%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.6 µs   1.2%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.9 µs   0.9%      1
void at::native::(anonymous namespace)::vectorized_layer_...       5.9 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              5.5 µs   0.9%      1
void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...       4.5 µs   0.7%      1
... 11 more
total                                                            637.5 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     443.2 µs  44.5%      2
void cutlass::device_kernel<flash::enable_sm90_or_later<f...     377.3 µs  37.9%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT                           36.2 µs   3.6%      1
nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT                          33.8 µs   3.4%      2
nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN                           22.7 µs   2.3%      1
_matmul                                                           10.8 µs   1.1%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       8.6 µs   0.9%      1
_finish                                                            8.3 µs   0.8%      1
void sglang::muse_spark_norm_residual_kernel<8192l, 512, ...       7.9 µs   0.8%      1
void at::native::(anonymous namespace)::vectorized_layer_...       6.4 µs   0.6%      1
_post_expert_norm_weighted_sum_kernel                              5.8 µs   0.6%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.7 µs   0.6%      1
... 11 more
total                                                            995.7 µs
GPU busy (kernels overlap: PDL or streams)                       981.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

