# moe_hybrid_m32_m128_v2

Job: `434b1d6c0336443e`
Correctness: True; official speedup: 1.97208x; CI: [1.9425026121035316, 2.0016651282082076]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 76.85 | 30.81 | 2.494x | 11.42 |
| decode 16 | experts (ep8) | 192.91 | 106.13 | 1.818x | 52.52 |
| decode 32 | experts (ep8) | 207.95 | 141.83 | 1.466x | 70.76 |
| verify 1 | experts (ep8) | 144.42 | 77.60 | 1.861x | 37.32 |
| verify 16 | experts (ep8) | 465.44 | 222.92 | 2.088x | 101.34 |
| verify 32 | experts (ep8) | 620.61 | 272.55 | 2.277x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.611 | 1.848 | 2.494x | 6.731x |
| decode 16 | 11.575 | 6.368 | 1.818x | 3.673x |
| decode 32 | 12.477 | 8.510 | 1.466x | 2.939x |
| verify 1 | 8.665 | 4.656 | 1.861x | 3.870x |
| verify 16 | 27.926 | 13.375 | 2.088x | 4.593x |
| verify 32 | 37.236 | 16.353 | 2.277x | 5.697x |

Recomputed geometric mean across loads: 1.972084x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      12.8 µs  47.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...       9.4 µs  34.6%      1
_post_expert_norm_weighted_sum_kernel                              3.0 µs  11.1%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.9 µs   7.0%      1
total                                                             27.1 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 8, 0, 0>(uint4 const*, uint2 const*...      59.5 µs  60.5%      1
void wm_gemm<1, 2, 2, 8, 0, 0>(uint4 const*, uint2 const*...      33.1 µs  33.7%      1
_post_expert_norm_weighted_sum_kernel                              4.0 µs   4.0%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.8 µs   1.8%      1
total                                                             98.3 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_om__w2                                                          151.5 µs  95.9%      2
_post_expert_norm_weighted_sum_kernel                              3.7 µs   2.3%      1
_om__route                                                         2.8 µs   1.8%      1
total                                                            158.0 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      31.6 µs  59.0%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      16.7 µs  31.2%      1
_post_expert_norm_weighted_sum_kernel                              3.5 µs   6.4%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.8 µs   3.3%      1
total                                                             53.6 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_om__w13_mix                                                     130.8 µs  62.1%      1
_om__w2_mix                                                       68.7 µs  32.6%      1
_om__route2                                                        6.4 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              4.7 µs   2.2%      1
total                                                            210.7 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_bucket<1, 2>(uint4 const*, uint2 const*, float co...     267.9 µs  28.0%      1
void wm_bucket<0, 2>(uint4 const*, uint2 const*, float co...     177.2 µs  18.5%      1
void wm_bucket<1, 4>(uint4 const*, uint2 const*, float co...     166.0 µs  17.4%      1
void wm_bucket<1, 1>(uint4 const*, uint2 const*, float co...     131.2 µs  13.7%      1
void wm_bucket<0, 1>(uint4 const*, uint2 const*, float co...     108.1 µs  11.3%      1
void wm_bucket<0, 4>(uint4 const*, uint2 const*, float co...      96.3 µs  10.1%      1
_post_expert_norm_weighted_sum_kernel                              5.5 µs   0.6%      1
route_kernel(int const*, int, int const*, int, int, int*,...       3.2 µs   0.3%      1
total                                                            955.4 µs
GPU busy (kernels overlap: PDL or streams)                       377.1 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

