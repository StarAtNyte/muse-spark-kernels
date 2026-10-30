# scored_prefill_boundary_256_hint_v1

Job: `9937a79b7f8448c4`
Correctness: True; official speedup: 1.76757x; CI: [1.7410602108656288, 1.7940874254097596]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| turn 1 | sliding | 1049.79 | 581.67 | 1.805x | 256.95 |
| turn 1 | full | 1414.45 | 906.10 | 1.561x | 496.69 |
| turn 4 | sliding | 3383.84 | 1813.28 | 1.866x | 1036.28 |
| turn 4 | full | 4843.33 | 3102.37 | 1.561x | 1992.42 |
| first turn | full | 16597.84 | 9137.44 | 1.816x | 5518.43 |
| first turn | sliding | 17389.55 | 9129.80 | 1.905x | 5663.35 |
| first turn | full | 21505.14 | 12723.63 | 1.690x | 7710.00 |
| first turn | full | 24096.98 | 15142.62 | 1.591x | 9847.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| turn 1 | 70.922 | 41.254 | 1.719x | 3.588x |
| turn 4 | 233.150 | 133.049 | 1.752x | 2.931x |
| first turn | 3394.957 | 1851.972 | 1.833x | 2.950x |

Recomputed geometric mean across loads: 1.767574x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### turn 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc_moe::moe_fused_persistent<0>(okc_moe::FusedArgs)        272.6 µs  46.1%      1
nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                   152.0 µs  25.7%      3
nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN                          39.2 µs   6.6%      1
void okc_moe::boundary_kernel<512, true, false, 0>(__nv_b...      21.4 µs   3.6%      1
void okc_attn::attn_fwd_kernel<3, 0, false, false, false,...      20.8 µs   3.5%      1
void okc_moe::boundary_kernel<512, false, false, 0>(__nv_...      19.1 µs   3.2%      1
void okc_attn::attn_combine2_kernel<2>(float const*, int,...      18.6 µs   3.2%      1
nvjet_sm90_tss_128x96_64x7_1x2_h_bz_splitK_TNT                    12.9 µs   2.2%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...       9.8 µs   1.7%      1
_combine_kernel                                                    9.2 µs   1.5%      1
okc_moe::moe_route_kernel(float const*, float const*, int...       5.3 µs   0.9%      1
void cublasLt::splitKreduce_kernel<32, 16, int, float, fl...       3.5 µs   0.6%      1
... 2 more
total                                                            590.7 µs
GPU busy (kernels overlap: PDL or streams)                       575.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### turn 1: full

```text
kernel                                                           GPU time  share  calls
void okc_attn::attn_fwd_kernel<3, 0, false, true, false, ...     346.8 µs  25.6%      1
void okc_attn::attn_combine2_kernel<2>(float const*, int,...     346.7 µs  25.6%      1
void okc_moe::moe_fused_persistent<0>(okc_moe::FusedArgs)        310.2 µs  22.9%      1
nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN                   150.5 µs  11.1%      3
nvjet_sm90_tss_128x96_64x7_1x2_h_bz_splitK_TNT                    77.0 µs   5.7%      1
nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN                          38.7 µs   2.9%      1
void okc_moe::boundary_kernel<512, true, false, 0>(__nv_b...      20.4 µs   1.5%      1
void okc_moe::boundary_kernel<512, false, false, 0>(__nv_...      18.5 µs   1.4%      1
okc_attn::attn_prep2_kernel(unsigned char const*, unsigne...      11.8 µs   0.9%      1
_combine_kernel                                                   10.0 µs   0.7%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...       9.8 µs   0.7%      1
okc_moe::moe_route_kernel(float const*, float const*, int...       6.2 µs   0.5%      1
... 2 more
total                                                           1353.2 µs
GPU busy (kernels overlap: PDL or streams)                       940.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### turn 4: sliding

```text
kernel                                                           GPU time  share  calls
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)        768.0 µs  42.3%      1
nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                   320.0 µs  17.6%      2
nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN                   255.0 µs  14.0%      1
nvjet_sm90_tst_192x144_64x5_1x2_h_bz_coopB_TNT                   149.9 µs   8.2%      1
void okc_moe::boundary_kernel<512, false, false, 0>(__nv_...      76.4 µs   4.2%      1
void okc_moe::boundary_kernel<512, true, false, 0>(__nv_b...      74.0 µs   4.1%      1
void okc_attn::attn_fwd_kernel<3, 0, false, false, false,...      61.7 µs   3.4%      1
nvjet_sm90_tss_128x104_64x7_4x2_h_bz_TNT                          38.2 µs   2.1%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...      27.1 µs   1.5%      1
_combine_kernel                                                   26.0 µs   1.4%      1
okc_moe::moe_route_kernel(float const*, float const*, int...       7.9 µs   0.4%      1
void okc_moe::qk_prep_kernel<true, false, long long, long...       6.6 µs   0.4%      1
... 2 more
total                                                           1817.6 µs
```

### turn 4: full

```text
kernel                                                           GPU time  share  calls
void okc_attn::attn_fwd_kernel<3, 0, false, true, false, ...    1352.3 µs  42.6%      1
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)        706.7 µs  22.3%      1
nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT                   322.0 µs  10.1%      2
nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN                   292.0 µs   9.2%      1
nvjet_sm90_tst_192x144_64x5_1x2_h_bz_coopB_TNT                   148.9 µs   4.7%      1
void okc_moe::boundary_kernel<512, false, false, 0>(__nv_...      75.8 µs   2.4%      1
void okc_moe::boundary_kernel<512, true, false, 0>(__nv_b...      75.6 µs   2.4%      1
okc_moe::moe_route_kernel(float const*, float const*, int...      58.5 µs   1.8%      1
okc_attn::attn_prep2_kernel(unsigned char const*, unsigne...      45.9 µs   1.4%      1
nvjet_sm90_tss_128x104_64x7_4x2_h_bz_TNT                          38.5 µs   1.2%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...      26.5 µs   0.8%      1
_combine_kernel                                                   24.5 µs   0.8%      1
... 2 more
total                                                           3175.5 µs
GPU busy (kernels overlap: PDL or streams)                      3065.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### first turn: full

```text
kernel                                                           GPU time  share  calls
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)       3322.8 µs  35.7%      1
nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN                        1660.5 µs  17.8%      2
nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT                        1468.6 µs  15.8%      1
void okc_attn::attn_fwd_kernel<3, 0, false, true, false, ...     833.7 µs   9.0%      1
nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN                   773.0 µs   8.3%      1
void okc_moe::boundary_kernel<512, true, false, 2>(__nv_b...     459.3 µs   4.9%      1
void okc_moe::boundary_kernel<256, false, true, 1>(__nv_b...     268.3 µs   2.9%      1
nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA_TNT                   177.1 µs   1.9%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...     126.8 µs   1.4%      1
_combine_kernel                                                  122.5 µs   1.3%      1
okc_moe::moe_route_kernel(float const*, float const*, int...      67.2 µs   0.7%      1
void okc_moe::qk_prep_kernel<false, false, long long, lon...      21.8 µs   0.2%      1
... 2 more
total                                                           9311.3 µs
GPU busy (kernels overlap: PDL or streams)                      9156.9 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### first turn: sliding

```text
kernel                                                           GPU time  share  calls
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)       3772.8 µs  41.1%      1
nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN                        1671.0 µs  18.2%      2
nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT                        1350.8 µs  14.7%      1
nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN                   773.1 µs   8.4%      1
void okc_moe::boundary_kernel<512, true, false, 2>(__nv_b...     461.6 µs   5.0%      1
void okc_attn::attn_fwd_kernel<3, 0, false, false, false,...     300.4 µs   3.3%      1
void okc_moe::boundary_kernel<256, false, true, 1>(__nv_b...     269.2 µs   2.9%      1
nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA_TNT                   235.9 µs   2.6%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...     140.6 µs   1.5%      1
_combine_kernel                                                  131.8 µs   1.4%      1
okc_moe::moe_route_kernel(float const*, float const*, int...      27.1 µs   0.3%      1
void okc_moe::qk_prep_kernel<true, false, long long, long...      24.2 µs   0.3%      1
... 2 more
total                                                           9170.5 µs
```

### first turn: full

```text
kernel                                                           GPU time  share  calls
void okc_attn::attn_fwd_kernel<3, 0, false, true, false, ...    3874.4 µs  30.8%      1
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)       3509.6 µs  27.9%      1
nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN                        1668.8 µs  13.3%      2
nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT                        1473.8 µs  11.7%      1
nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN                   771.5 µs   6.1%      1
void okc_moe::boundary_kernel<512, true, false, 2>(__nv_b...     464.0 µs   3.7%      1
void okc_moe::boundary_kernel<256, false, true, 1>(__nv_b...     269.5 µs   2.1%      1
nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA_TNT                   178.5 µs   1.4%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...     128.4 µs   1.0%      1
_combine_kernel                                                  125.8 µs   1.0%      1
okc_moe::moe_route_kernel(float const*, float const*, int...      67.3 µs   0.5%      1
void okc_moe::qk_prep_kernel<false, false, long long, lon...      22.1 µs   0.2%      1
... 2 more
total                                                          12571.1 µs
GPU busy (kernels overlap: PDL or streams)                     12408.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### first turn: full

```text
kernel                                                           GPU time  share  calls
void okc_attn::attn_fwd_kernel<3, 0, false, true, false, ...    6815.0 µs  43.9%      1
void okc_moe::moe_fused_persistent<1>(okc_moe::FusedArgs)       3531.8 µs  22.7%      1
nvjet_sm90_tst_176x128_64x5_1x2_h_bz_TNN                        1666.2 µs  10.7%      2
nvjet_sm90_tst_128x160_64x5_2x1_v_bz_TNT                        1471.8 µs   9.5%      1
nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN                   772.7 µs   5.0%      1
void okc_moe::boundary_kernel<512, true, false, 2>(__nv_b...     460.7 µs   3.0%      1
void okc_moe::boundary_kernel<256, false, true, 1>(__nv_b...     270.0 µs   1.7%      1
nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA_TNT                   178.0 µs   1.1%      1
void okc_moe::moe_gather_kernel<true>(__nv_bfloat16 const...     133.8 µs   0.9%      1
_combine_kernel                                                  127.4 µs   0.8%      1
okc_moe::moe_route_kernel(float const*, float const*, int...      67.0 µs   0.4%      1
void okc_moe::qk_prep_kernel<false, false, long long, lon...      22.2 µs   0.1%      1
... 2 more
total                                                          15535.8 µs
GPU busy (kernels overlap: PDL or streams)                     15371.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

