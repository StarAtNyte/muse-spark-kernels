# scored_decoder_compact_qkv_o_v1

Job: `b72f02c7eb8745f5`
Correctness: True; official speedup: 2.26032x; CI: [2.226416069262645, 2.2942256957376492]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 172.35 | 68.11 | 2.530x | 40.02 |
| decode 1 | full | 180.51 | 74.58 | 2.420x | 44.18 |
| decode 16 | sliding | 264.80 | 127.79 | 2.072x | 76.72 |
| decode 16 | full | 443.74 | 206.25 | 2.151x | 137.46 |
| decode 32 | sliding | 271.04 | 148.24 | 1.828x | 84.51 |
| decode 32 | full | 639.34 | 292.06 | 2.189x | 208.10 |
| verify 1 | sliding | 265.46 | 107.26 | 2.475x | 62.29 |
| verify 1 | full | 277.76 | 114.39 | 2.428x | 66.44 |
| verify 16 | sliding | 529.84 | 237.47 | 2.231x | 126.61 |
| verify 16 | full | 716.61 | 324.60 | 2.208x | 188.68 |
| verify 32 | sliding | 741.79 | 311.13 | 2.384x | 143.81 |
| verify 32 | full | 1073.18 | 470.73 | 2.280x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.816 | 4.326 | 2.500x | 4.245x |
| decode 16 | 19.281 | 9.179 | 2.101x | 3.366x |
| decode 32 | 22.697 | 11.492 | 1.975x | 3.145x |
| verify 1 | 16.655 | 6.764 | 2.462x | 4.240x |
| verify 16 | 35.838 | 16.117 | 2.224x | 4.053x |
| verify 32 | 51.293 | 21.844 | 2.348x | 4.710x |

Recomputed geometric mean across loads: 2.260321x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           29.4 µs  38.0%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      17.9 µs  23.0%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...       9.9 µs  12.8%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.0 µs   7.8%      1
void (anonymous namespace)::attn_kernel<1, false, 13>((an...       4.9 µs   6.4%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.7 µs   6.0%      1
_gle_route                                                         2.7 µs   3.5%      1
_sl_cluster_finish                                                 2.0 µs   2.5%      1
total                                                             77.5 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           21.9 µs  28.7%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.0 µs  23.6%      1
void (anonymous namespace)::attn_kernel<1, false, 11>((an...      10.9 µs  14.3%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.2 µs  13.4%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.2 µs   8.1%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.6 µs   6.1%      1
_gle_route                                                         2.6 µs   3.4%      1
_sl_cluster_finish                                                 1.7 µs   2.3%      1
total                                                             76.3 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           58.0 µs  46.9%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      18.5 µs  15.0%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      15.1 µs  12.2%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...       9.9 µs   8.0%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       7.6 µs   6.2%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       6.0 µs   4.8%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.4 µs   2.8%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.1 µs   2.5%      1
_sl_tile_route                                                     2.0 µs   1.6%      1
total                                                            123.6 µs
GPU busy (kernels overlap: PDL or streams)                       121.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...      86.4 µs  41.2%      1
void okc::moe_fused<false>(okc::Params)                           64.9 µs  31.0%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      17.1 µs   8.2%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      14.6 µs   7.0%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...       9.8 µs   4.7%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       7.6 µs   3.6%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.6 µs   1.7%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.6 µs   1.7%      1
_sl_tile_route                                                     2.0 µs   1.0%      1
total                                                            209.6 µs
GPU busy (kernels overlap: PDL or streams)                       204.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           92.0 µs  54.4%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      17.3 µs  10.3%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      16.5 µs   9.8%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.4 µs   6.7%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       9.4 µs   5.6%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       7.6 µs   4.5%      1
_graft_row_boundary_pre                                            7.4 µs   4.4%      2
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.4 µs   3.2%      1
_sl_tile_route                                                     2.1 µs   1.2%      1
total                                                            169.2 µs
GPU busy (kernels overlap: PDL or streams)                       166.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...     162.5 µs  52.3%      1
void okc::moe_fused<false>(okc::Params)                           79.1 µs  25.5%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      18.1 µs   5.8%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      16.5 µs   5.3%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.3 µs   3.6%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       8.1 µs   2.6%      1
_graft_row_boundary_pre                                            7.8 µs   2.5%      2
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.2 µs   1.7%      1
_sl_tile_route                                                     2.2 µs   0.7%      1
total                                                            310.8 µs
GPU busy (kernels overlap: PDL or streams)                       303.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           57.2 µs  51.3%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      20.7 µs  18.6%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...       9.2 µs   8.3%      1
void (anonymous namespace)::attn_kernel<8, false, 15>((an...       6.9 µs   6.2%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       5.9 µs   5.3%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.8 µs   4.3%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   2.9%      1
_sl_tile_route                                                     1.9 µs   1.7%      1
_sl_cluster_finish                                                 1.8 µs   1.6%      1
total                                                            111.6 µs
GPU busy (kernels overlap: PDL or streams)                       110.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           64.3 µs  50.4%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.7 µs  15.5%      1
void (anonymous namespace)::attn_kernel<8, false, 11>((an...      14.4 µs  11.3%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...      10.4 µs   8.2%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.1 µs   4.8%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       5.2 µs   4.1%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   2.5%      1
_sl_cluster_finish                                                 2.2 µs   1.7%      1
_sl_tile_route                                                     1.9 µs   1.5%      1
total                                                            127.6 µs
GPU busy (kernels overlap: PDL or streams)                       125.8 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           160.3 µs  63.7%      1
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.4 µs  10.1%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           21.1 µs   8.4%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.0 µs   5.6%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      11.7 µs   4.7%      1
_graft_row_boundary                                                9.2 µs   3.6%      2
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.4 µs   2.9%      1
_graft_route                                                       2.8 µs   1.1%      1
total                                                            251.7 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           166.4 µs  47.8%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     101.8 µs  29.2%      1
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     24.6 µs   7.1%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           21.4 µs   6.2%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.5 µs   4.2%      1
_graft_row_boundary                                                9.1 µs   2.6%      2
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.5 µs   2.1%      1
_graft_route                                                       2.8 µs   0.8%      1
total                                                            348.0 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           199.0 µs  62.0%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.2 µs   9.7%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    27.4 µs   8.5%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           21.9 µs   6.8%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      15.4 µs   4.8%      1
_graft_row_boundary                                               13.6 µs   4.2%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     8.9 µs   2.8%      1
_graft_route                                                       3.5 µs   1.1%      1
total                                                            320.8 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           194.7 µs  40.6%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     181.4 µs  37.8%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.2 µs   6.5%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    25.9 µs   5.4%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           20.1 µs   4.2%      1
_graft_row_boundary                                               13.5 µs   2.8%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     9.4 µs   2.0%      1
_graft_route                                                       3.5 µs   0.7%      1
total                                                            479.7 µs
```

