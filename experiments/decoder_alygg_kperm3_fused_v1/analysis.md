# decoder_alygg_kperm3_fused_v1

Job: `13161eab55cb447a`
Correctness: True; official speedup: 2.20175x; CI: [2.1687203963233923, 2.2347727941809574]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 170.67 | 68.18 | 2.503x | 40.02 |
| decode 1 | full | 181.90 | 75.17 | 2.420x | 44.18 |
| decode 16 | sliding | 264.98 | 128.02 | 2.070x | 76.72 |
| decode 16 | full | 441.38 | 209.85 | 2.103x | 137.46 |
| decode 32 | sliding | 261.70 | 153.30 | 1.707x | 84.51 |
| decode 32 | full | 615.57 | 293.33 | 2.099x | 208.10 |
| verify 1 | sliding | 259.62 | 105.02 | 2.472x | 62.29 |
| verify 1 | full | 281.44 | 117.95 | 2.386x | 66.44 |
| verify 16 | sliding | 517.63 | 246.37 | 2.101x | 126.61 |
| verify 16 | full | 681.34 | 325.35 | 2.094x | 188.68 |
| verify 32 | sliding | 721.66 | 310.08 | 2.327x | 143.81 |
| verify 32 | full | 1077.07 | 476.38 | 2.261x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.761 | 4.339 | 2.480x | 4.223x |
| decode 16 | 19.251 | 9.247 | 2.082x | 3.361x |
| decode 32 | 21.887 | 11.745 | 1.864x | 3.033x |
| verify 1 | 16.445 | 6.718 | 2.448x | 4.186x |
| verify 16 | 34.713 | 16.539 | 2.099x | 3.925x |
| verify 32 | 50.430 | 21.886 | 2.304x | 4.631x |

Recomputed geometric mean across loads: 2.201747x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           22.9 µs  31.5%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.1 µs  26.2%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.4 µs  14.3%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.5 µs   9.0%      1
void (anonymous namespace)::attn_kernel<1, false, 13>((an...       4.8 µs   6.6%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.5 µs   6.3%      1
_gle_route                                                         2.7 µs   3.7%      1
_sl_cluster_finish                                                 1.7 µs   2.4%      1
total                                                             72.7 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           22.0 µs  28.7%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.4 µs  24.0%      1
void (anonymous namespace)::attn_kernel<1, false, 11>((an...      10.8 µs  14.1%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.3 µs  13.4%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.3 µs   8.3%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.4 µs   5.7%      1
_gle_route                                                         2.5 µs   3.3%      1
_sl_cluster_finish                                                 1.8 µs   2.4%      1
total                                                             76.7 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           65.8 µs  50.0%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      18.3 µs  13.9%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      15.3 µs  11.6%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      10.0 µs   7.6%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       7.6 µs   5.7%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       6.1 µs   4.6%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.5 µs   2.7%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.2 µs   2.4%      1
_sl_tile_route                                                     2.0 µs   1.5%      1
total                                                            131.7 µs
GPU busy (kernels overlap: PDL or streams)                       129.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...      89.9 µs  41.8%      1
void okc::moe_fused<false>(okc::Params)                           65.4 µs  30.5%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      17.3 µs   8.0%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      14.8 µs   6.9%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      10.0 µs   4.6%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       7.9 µs   3.7%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.8 µs   1.8%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.6 µs   1.7%      1
_sl_tile_route                                                     2.1 µs   1.0%      1
total                                                            214.8 µs
GPU busy (kernels overlap: PDL or streams)                       209.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           80.6 µs  49.8%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      18.1 µs  11.2%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      16.4 µs  10.2%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.9 µs   7.3%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       9.2 µs   5.7%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       6.8 µs   4.2%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.3 µs   3.3%      1
_boundary2                                                         5.1 µs   3.2%      1
_boundary1                                                         4.5 µs   2.8%      1
_route_split                                                       4.0 µs   2.5%      1
total                                                            161.8 µs
GPU busy (kernels overlap: PDL or streams)                       158.7 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...     169.7 µs  52.4%      1
void okc::moe_fused<false>(okc::Params)                           79.9 µs  24.7%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      17.8 µs   5.5%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      17.7 µs   5.5%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.1 µs   3.4%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       8.5 µs   2.6%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.4 µs   1.7%      1
_boundary1                                                         4.9 µs   1.5%      1
_boundary2                                                         4.8 µs   1.5%      1
_route_split                                                       4.1 µs   1.3%      1
total                                                            323.8 µs
GPU busy (kernels overlap: PDL or streams)                       313.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           64.4 µs  54.3%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.5 µs  16.5%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...       9.3 µs   7.9%      1
void (anonymous namespace)::attn_kernel<8, false, 15>((an...       7.1 µs   6.0%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.1 µs   5.1%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       5.3 µs   4.5%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   2.7%      1
_sl_cluster_finish                                                 1.8 µs   1.5%      1
_sl_tile_route                                                     1.8 µs   1.5%      1
total                                                            118.5 µs
GPU busy (kernels overlap: PDL or streams)                       116.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           56.7 µs  47.7%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.8 µs  16.6%      1
void (anonymous namespace)::attn_kernel<8, false, 11>((an...      14.7 µs  12.4%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...       9.5 µs   8.0%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.2 µs   5.3%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       5.0 µs   4.2%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   2.7%      1
_sl_tile_route                                                     2.0 µs   1.6%      1
_sl_cluster_finish                                                 1.8 µs   1.5%      1
total                                                            118.8 µs
GPU busy (kernels overlap: PDL or streams)                       117.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           141.6 µs  60.0%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           28.6 µs  12.1%      2
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.0 µs  10.6%      1
nvjet_sm90_tst_48x64_64x15_1x2_h_bz_TNN                           16.6 µs   7.0%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      11.9 µs   5.1%      1
_graft_row_boundary                                                9.7 µs   4.1%      2
_graft_route                                                       2.6 µs   1.1%      1
total                                                            236.0 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           145.8 µs  43.7%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     104.5 µs  31.3%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           28.9 µs   8.7%      2
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.6 µs   7.7%      1
nvjet_sm90_tst_48x64_64x15_1x2_h_bz_TNN                           17.5 µs   5.2%      1
_graft_row_boundary                                                9.0 µs   2.7%      2
_graft_route                                                       2.6 µs   0.8%      1
total                                                            334.0 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           188.1 µs  61.2%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.2 µs  10.2%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    26.4 µs   8.6%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.9 µs   6.5%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      15.2 µs   4.9%      1
_graft_row_boundary                                               14.3 µs   4.6%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     8.9 µs   2.9%      1
_graft_route                                                       3.5 µs   1.1%      1
total                                                            307.4 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           206.4 µs  41.3%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     189.4 µs  37.9%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           32.7 µs   6.5%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    25.6 µs   5.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.6 µs   4.1%      1
_graft_row_boundary                                               13.2 µs   2.6%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     8.9 µs   1.8%      1
_graft_route                                                       3.4 µs   0.7%      1
total                                                            500.3 µs
```

