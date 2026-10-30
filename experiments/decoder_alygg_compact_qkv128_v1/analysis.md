# decoder_alygg_compact_qkv128_v1

Job: `9a3e2477765a4b30`
Correctness: True; official speedup: 2.21377x; CI: [2.1805596352030068, 2.246972619016296]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 171.58 | 68.36 | 2.510x | 40.02 |
| decode 1 | full | 180.19 | 74.86 | 2.407x | 44.18 |
| decode 16 | sliding | 257.81 | 126.05 | 2.045x | 76.72 |
| decode 16 | full | 435.47 | 207.52 | 2.098x | 137.46 |
| decode 32 | sliding | 270.54 | 154.82 | 1.747x | 84.51 |
| decode 32 | full | 626.00 | 306.34 | 2.043x | 208.10 |
| verify 1 | sliding | 266.46 | 109.36 | 2.436x | 62.29 |
| verify 1 | full | 273.81 | 113.18 | 2.419x | 66.44 |
| verify 16 | sliding | 539.09 | 248.42 | 2.170x | 126.61 |
| verify 16 | full | 736.82 | 343.20 | 2.147x | 188.68 |
| verify 32 | sliding | 759.97 | 321.10 | 2.367x | 143.81 |
| verify 32 | full | 1115.18 | 488.18 | 2.284x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.776 | 4.342 | 2.482x | 4.229x |
| decode 16 | 18.827 | 9.119 | 2.065x | 3.287x |
| decode 32 | 22.461 | 12.023 | 1.868x | 3.112x |
| verify 1 | 16.638 | 6.842 | 2.432x | 4.236x |
| verify 16 | 36.587 | 16.918 | 2.163x | 4.137x |
| verify 32 | 52.801 | 22.581 | 2.338x | 4.848x |

Recomputed geometric mean across loads: 2.213766x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           22.9 µs  31.5%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.7 µs  25.7%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.4 µs  14.3%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.4 µs   8.8%      1
void (anonymous namespace)::attn_kernel<1, false, 13>((an...       4.9 µs   6.7%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.9 µs   6.7%      1
_gle_route                                                         2.6 µs   3.5%      1
_sl_cluster_finish                                                 2.0 µs   2.8%      1
total                                                             72.8 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           28.5 µs  34.3%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.5 µs  22.3%      1
void (anonymous namespace)::attn_kernel<1, false, 11>((an...      10.9 µs  13.1%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.0 µs  12.0%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.2 µs   7.5%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.4 µs   5.4%      1
_gle_route                                                         2.4 µs   2.9%      1
_sl_cluster_finish                                                 2.0 µs   2.4%      1
total                                                             83.0 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           81.8 µs  55.0%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      18.2 µs  12.2%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      15.6 µs  10.5%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      10.9 µs   7.3%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       7.7 µs   5.2%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       6.0 µs   4.1%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.5 µs   2.3%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.1 µs   2.1%      1
_sl_tile_route                                                     2.0 µs   1.3%      1
total                                                            148.8 µs
GPU busy (kernels overlap: PDL or streams)                       146.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...      90.9 µs  40.3%      1
void okc::moe_fused<false>(okc::Params)                           73.2 µs  32.4%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      17.5 µs   7.8%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      15.5 µs   6.9%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      11.0 µs   4.9%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       7.9 µs   3.5%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.8 µs   1.7%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.6 µs   1.6%      1
_sl_tile_route                                                     2.1 µs   0.9%      1
total                                                            225.5 µs
GPU busy (kernels overlap: PDL or streams)                       219.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           79.1 µs  50.6%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      18.4 µs  11.8%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      16.5 µs  10.5%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.8 µs   7.5%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       9.1 µs   5.8%      1
_graft_row_boundary_pre                                            7.4 µs   4.7%      2
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       6.7 µs   4.3%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.5 µs   3.5%      1
_sl_tile_route                                                     2.0 µs   1.3%      1
total                                                            156.5 µs
GPU busy (kernels overlap: PDL or streams)                       153.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...     169.9 µs  54.4%      1
void okc::moe_fused<false>(okc::Params)                           74.8 µs  24.0%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      17.6 µs   5.6%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      15.2 µs   4.9%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.9 µs   3.8%      1
_graft_row_boundary_pre                                            7.7 µs   2.5%      2
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       7.6 µs   2.4%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.3 µs   1.7%      1
_sl_tile_route                                                     2.0 µs   0.7%      1
total                                                            312.1 µs
GPU busy (kernels overlap: PDL or streams)                       305.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           52.9 µs  49.6%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.3 µs  18.1%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...       9.4 µs   8.8%      1
void (anonymous namespace)::attn_kernel<8, false, 15>((an...       7.2 µs   6.7%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.1 µs   5.7%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       5.0 µs   4.7%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   3.0%      1
_sl_tile_route                                                     1.9 µs   1.8%      1
_sl_cluster_finish                                                 1.7 µs   1.6%      1
total                                                            106.7 µs
GPU busy (kernels overlap: PDL or streams)                       105.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           43.6 µs  40.7%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      20.2 µs  18.8%      1
void (anonymous namespace)::attn_kernel<8, false, 11>((an...      14.8 µs  13.8%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...      10.7 µs  10.0%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       5.9 µs   5.5%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.9 µs   4.6%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.4 µs   3.2%      1
_sl_tile_route                                                     1.9 µs   1.8%      1
_sl_cluster_finish                                                 1.8 µs   1.7%      1
total                                                            107.1 µs
GPU busy (kernels overlap: PDL or streams)                       105.6 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           149.1 µs  61.0%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           28.9 µs  11.8%      2
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.4 µs  10.4%      1
nvjet_sm90_tst_48x64_64x15_1x2_h_bz_TNN                           17.2 µs   7.0%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      12.0 µs   4.9%      1
_graft_row_boundary                                                9.0 µs   3.7%      2
_graft_route                                                       2.6 µs   1.1%      1
total                                                            244.3 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           147.2 µs  44.0%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     103.0 µs  30.8%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           29.8 µs   8.9%      2
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.8 µs   7.7%      1
nvjet_sm90_tst_48x64_64x15_1x2_h_bz_TNN                           17.1 µs   5.1%      1
_graft_row_boundary                                                9.0 µs   2.7%      2
_graft_route                                                       2.6 µs   0.8%      1
total                                                            334.4 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           201.7 µs  63.1%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.2 µs   9.8%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    26.0 µs   8.1%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           19.8 µs   6.2%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      15.2 µs   4.8%      1
_graft_row_boundary                                               13.6 µs   4.2%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     8.9 µs   2.8%      1
_graft_route                                                       3.4 µs   1.1%      1
total                                                            319.8 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           217.0 µs  42.7%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     186.0 µs  36.6%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.4 µs   6.2%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    26.5 µs   5.2%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.8 µs   4.1%      1
_graft_row_boundary                                               13.6 µs   2.7%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     9.2 µs   1.8%      1
_graft_route                                                       3.4 µs   0.7%      1
total                                                            507.8 µs
```

