# decoder_alygg_forced_qkv_o_fixed_v1

Job: `851e490e09f44621`
Correctness: True; official speedup: 2.19920x; CI: [2.1662122939946, 2.23218830294875]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 174.18 | 70.40 | 2.474x | 40.02 |
| decode 1 | full | 183.41 | 75.87 | 2.417x | 44.18 |
| decode 16 | sliding | 257.74 | 122.90 | 2.097x | 76.72 |
| decode 16 | full | 437.02 | 206.81 | 2.113x | 137.46 |
| decode 32 | sliding | 262.43 | 147.78 | 1.776x | 84.51 |
| decode 32 | full | 615.50 | 303.45 | 2.028x | 208.10 |
| verify 1 | sliding | 270.78 | 113.79 | 2.380x | 62.29 |
| verify 1 | full | 274.03 | 114.59 | 2.391x | 66.44 |
| verify 16 | sliding | 509.36 | 235.49 | 2.163x | 126.61 |
| verify 16 | full | 701.60 | 328.11 | 2.138x | 188.68 |
| verify 32 | sliding | 697.50 | 304.35 | 2.292x | 143.81 |
| verify 32 | full | 1034.94 | 467.17 | 2.215x | 267.18 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 10.947 | 4.452 | 2.459x | 4.296x |
| decode 16 | 18.849 | 8.962 | 2.103x | 3.290x |
| decode 32 | 21.920 | 11.653 | 1.881x | 3.037x |
| verify 1 | 16.841 | 7.068 | 2.383x | 4.287x |
| verify 16 | 34.656 | 16.082 | 2.155x | 3.919x |
| verify 32 | 48.644 | 21.475 | 2.265x | 4.467x |

Recomputed geometric mean across loads: 2.199200x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           30.1 µs  38.1%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.1 µs  23.0%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.2 µs  12.9%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.4 µs   8.1%      1
void (anonymous namespace)::attn_kernel<1, false, 13>((an...       5.0 µs   6.3%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.7 µs   5.9%      1
_gle_route                                                         2.6 µs   3.3%      1
_sl_cluster_finish                                                 1.9 µs   2.4%      1
total                                                             79.0 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           28.5 µs  34.2%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      18.0 µs  21.7%      1
void (anonymous namespace)::attn_kernel<1, false, 11>((an...      11.1 µs  13.3%      1
void sk2_kernel<1, 8, 3, 1, 4, 2, 6, true, false, 1, 1, _...      10.4 µs  12.5%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 4, 1, 4, true...       6.5 µs   7.8%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.5 µs   5.5%      1
_gle_route                                                         2.5 µs   3.0%      1
_sl_cluster_finish                                                 1.7 µs   2.0%      1
total                                                             83.4 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           65.9 µs  49.7%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      18.9 µs  14.3%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      14.9 µs  11.2%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      10.5 µs   7.9%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       7.5 µs   5.6%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       6.2 µs   4.7%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.6 µs   2.7%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.2 µs   2.4%      1
_sl_tile_route                                                     2.0 µs   1.5%      1
total                                                            132.7 µs
GPU busy (kernels overlap: PDL or streams)                       130.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...      90.7 µs  41.7%      1
void okc::moe_fused<false>(okc::Params)                           65.6 µs  30.1%      1
void sk2_kernel<1, 16, 4, 1, 8, 1, 5, true, false, 2, 1, ...      17.8 µs   8.2%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 5, true, false, 2, 1, ...      15.2 µs   7.0%      1
void sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 1, 1, ...      10.6 µs   4.8%      1
void sk2_kernel<1, 16, 4, 1, 4, 1, 4, true, false, 2, 1, ...       8.3 µs   3.8%      1
void glue_cluster_boundary<4, 32, true, false, false, tru...       3.8 µs   1.8%      1
void glue_cluster_boundary<4, 32, false, false, false, tr...       3.6 µs   1.6%      1
_sl_tile_route                                                     2.1 µs   1.0%      1
total                                                            217.8 µs
GPU busy (kernels overlap: PDL or streams)                       211.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           81.6 µs  50.8%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      18.5 µs  11.5%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      16.3 µs  10.1%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      12.3 µs   7.7%      1
void (anonymous namespace)::attn_kernel<1, false, 1>((ano...       9.3 µs   5.8%      1
_graft_row_boundary_pre                                            7.6 µs   4.8%      2
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       6.7 µs   4.2%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       6.2 µs   3.9%      1
_sl_tile_route                                                     2.0 µs   1.3%      1
total                                                            160.6 µs
GPU busy (kernels overlap: PDL or streams)                       157.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<1, true, 0>((anon...     170.2 µs  52.8%      1
void okc::moe_fused<false>(okc::Params)                           80.8 µs  25.1%      1
void sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, ...      18.5 µs   5.7%      1
void sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, ...      17.4 µs   5.4%      1
void sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, ...      11.6 µs   3.6%      1
_graft_row_boundary_pre                                            8.4 µs   2.6%      2
void sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, ...       7.8 µs   2.4%      1
void sk2_kernel<2, 32, 1, 1, 8, 1, 4, false, true, 4, 3, ...       5.3 µs   1.7%      1
_sl_tile_route                                                     2.1 µs   0.7%      1
total                                                            322.3 µs
GPU busy (kernels overlap: PDL or streams)                       312.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           57.4 µs  51.2%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.0 µs  16.9%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...       9.8 µs   8.8%      1
void (anonymous namespace)::attn_kernel<8, false, 15>((an...       6.9 µs   6.2%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       6.3 µs   5.6%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.0 µs   5.4%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.2 µs   2.8%      1
_sl_tile_route                                                     1.8 µs   1.6%      1
_sl_cluster_finish                                                 1.8 µs   1.6%      1
total                                                            112.2 µs
GPU busy (kernels overlap: PDL or streams)                       110.3 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           47.1 µs  43.0%      1
void sk2_kernel_gle35a06ad35022<1, 8, 4, 1, 8, 1, 6, true...      19.8 µs  18.0%      1
void (anonymous namespace)::attn_kernel<8, false, 11>((an...      14.5 µs  13.2%      1
void sk2_kernel_narrow_12cf58219a16<1, 8, 3, 1, 4, 2, 5, ...      10.2 µs   9.3%      1
void sk2_kernel<1, 8, 4, 1, 4, 1, 4, true, false, 2, 1, _...       6.1 µs   5.6%      1
void sk2_kernel<1, 8, 1, 1, 8, 2, 4, true, false, 2, 1, _...       4.8 µs   4.4%      1
void glue_cluster_boundary<4, 32, false, false, true, tru...       3.3 µs   3.0%      1
_sl_tile_route                                                     1.9 µs   1.8%      1
_sl_cluster_finish                                                 1.9 µs   1.7%      1
total                                                            109.6 µs
GPU busy (kernels overlap: PDL or streams)                       108.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           162.0 µs  63.6%      1
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     26.0 µs  10.2%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           21.6 µs   8.5%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.1 µs   5.5%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      11.8 µs   4.6%      1
_graft_row_boundary                                                9.5 µs   3.7%      2
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.1 µs   2.8%      1
_graft_route                                                       2.7 µs   1.1%      1
total                                                            254.7 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           140.8 µs  43.5%      1
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     103.3 µs  31.9%      1
nvjet_sm90_tss_80x64_64x11_4x2_h_bz_coopB_TNN                     25.0 µs   7.7%      1
nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT                           21.2 µs   6.6%      1
void qb_4d509bcf5bddaa48::qb_kernel<64, 36, 128, 3, 8, 53...      14.4 µs   4.4%      1
_graft_row_boundary                                                9.0 µs   2.8%      2
void ol_projection<128, 128, 4, 3, 1, 1, 1>(CUtensorMap_s...       7.1 µs   2.2%      1
_graft_route                                                       2.7 µs   0.8%      1
total                                                            323.4 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           190.0 µs  61.2%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           32.2 µs  10.4%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    25.8 µs   8.3%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           21.3 µs   6.9%      1
void (anonymous namespace)::attn_kernel<8, false, 3>((ano...      14.9 µs   4.8%      1
_graft_row_boundary                                               13.9 µs   4.5%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     8.8 µs   2.8%      1
_graft_route                                                       3.5 µs   1.1%      1
total                                                            310.3 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
void (anonymous namespace)::attn_kernel<8, true, 0>((anon...     187.4 µs  39.5%      1
void okc::moe_fused<true>(okc::Params)                           182.7 µs  38.5%      1
nvjet_sm90_tss_144x64_64x8_1x1_v_bz_TNN                           31.9 µs   6.7%      1
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT                    25.8 µs   5.4%      1
nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN                           21.6 µs   4.5%      1
_graft_row_boundary                                               13.2 µs   2.8%      2
nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNN                     9.0 µs   1.9%      1
_graft_route                                                       3.4 µs   0.7%      1
total                                                            474.8 µs
```

