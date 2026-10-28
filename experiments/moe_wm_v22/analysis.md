# moe_wm_v22

Job: `3fa3bbfe4a10455f`
Correctness: True; official speedup: 1.91656x; CI: [1.8878078266309009, 1.9453045117059533]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 77.12 | 31.16 | 2.475x | 11.42 |
| decode 16 | experts (ep8) | 187.06 | 103.19 | 1.813x | 52.52 |
| decode 32 | experts (ep8) | 203.84 | 151.85 | 1.342x | 70.76 |
| verify 1 | experts (ep8) | 142.61 | 76.65 | 1.861x | 37.32 |
| verify 16 | experts (ep8) | 473.55 | 228.94 | 2.068x | 101.34 |
| verify 32 | experts (ep8) | 614.56 | 287.40 | 2.138x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.627 | 1.870 | 2.475x | 6.755x |
| decode 16 | 11.223 | 6.191 | 1.813x | 3.562x |
| decode 32 | 12.230 | 9.111 | 1.342x | 2.881x |
| verify 1 | 8.556 | 4.599 | 1.861x | 3.821x |
| verify 16 | 28.413 | 13.737 | 2.068x | 4.673x |
| verify 32 | 36.874 | 17.244 | 2.138x | 5.641x |

Recomputed geometric mean across loads: 1.916556x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      20.4 µs  49.3%      1
void wm_gemm<1, 2, 1, 4, 0, 0>(uint4 const*, uint2 const*...      16.0 µs  38.7%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   7.7%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.8 µs   4.4%      1
total                                                             41.4 µs
GPU busy (kernels overlap: PDL or streams)                        36.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 2, 8, 0, 0>(uint4 const*, uint2 const*...      80.5 µs  62.9%      1
void wm_gemm<1, 2, 2, 8, 0, 0>(uint4 const*, uint2 const*...      40.6 µs  31.7%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   3.9%      1
route_kernel(int const*, int, int const*, int, int, int*,...       1.9 µs   1.5%      1
total                                                            127.9 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...     107.3 µs  60.3%      1
void wm_gemm<1, 4, 4, 4, 1, 0>(uint4 const*, uint2 const*...      64.1 µs  36.0%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   2.6%      1
route_kernel(int const*, int, int const*, int, int, int*,...       2.0 µs   1.1%      1
total                                                            178.0 µs
GPU busy (kernels overlap: PDL or streams)                       174.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm<0, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      56.8 µs  57.3%      1
void wm_gemm<1, 2, 1, 2, 0, 0>(uint4 const*, uint2 const*...      36.9 µs  37.2%      1
_post_expert_norm_weighted_sum_kernel                              3.5 µs   3.5%      1
route_kernel(int const*, int, int const*, int, int, int*,...       2.0 µs   2.0%      1
total                                                             99.1 µs
GPU busy (kernels overlap: PDL or streams)                        96.4 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     129.9 µs  51.2%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     117.2 µs  46.2%      1
_post_expert_norm_weighted_sum_kernel                              4.3 µs   1.7%      1
route_kernel(int const*, int, int const*, int, int, int*,...       2.1 µs   0.8%      1
total                                                            253.4 µs
GPU busy (kernels overlap: PDL or streams)                       215.2 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void wm_gemm_rs<0, 4, 4, 4, 0>(uint4 const*, uint2 const*...     150.6 µs  55.4%      1
void wm_gemm_rs<1, 4, 4, 4, 0>(uint4 const*, uint2 const*...     113.8 µs  41.9%      1
_post_expert_norm_weighted_sum_kernel                              4.6 µs   1.7%      1
route_kernel(int const*, int, int const*, int, int, int*,...       2.7 µs   1.0%      1
total                                                            271.7 µs
GPU busy (kernels overlap: PDL or streams)                       249.5 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
```

