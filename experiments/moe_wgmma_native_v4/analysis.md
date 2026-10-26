# moe_wgmma_native_v4

Job: `769e726bc944456a`
Correctness: True; official speedup: 0.58072x; CI: [0.5720044851411867, 0.5894259415414259]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 77.79 | 77.21 | 1.008x | 11.42 |
| decode 16 | experts (ep8) | 188.48 | 188.06 | 1.002x | 52.52 |
| decode 32 | experts (ep8) | 205.39 | 205.12 | 1.001x | 70.76 |
| verify 1 | experts (ep8) | 151.06 | 150.81 | 1.002x | 37.32 |
| verify 16 | experts (ep8) | 456.30 | 2554.20 | 0.179x | 101.34 |
| verify 32 | experts (ep8) | 623.90 | 2943.51 | 0.212x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.668 | 4.632 | 1.008x | 6.814x |
| decode 16 | 11.309 | 11.284 | 1.002x | 3.589x |
| decode 32 | 12.324 | 12.307 | 1.001x | 2.903x |
| verify 1 | 9.063 | 9.048 | 1.002x | 4.048x |
| verify 16 | 27.378 | 153.252 | 0.179x | 4.503x |
| verify 32 | 37.434 | 176.610 | 0.212x | 5.727x |

Recomputed geometric mean across loads: 0.580715x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      40.6 µs  53.8%      2
void at::native::unrolled_elementwise_kernel<at::native::...       5.8 µs   7.6%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       5.5 µs   7.2%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       4.9 µs   6.5%      3
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.5 µs   5.9%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       2.7 µs   3.6%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.8 µs   2.4%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   1.9%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.3 µs   1.7%      1
_moe_align_small_numel_kernel                                      1.3 µs   1.7%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.1 µs   1.5%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.0 µs   1.4%      1
... 4 more
total                                                             75.5 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     149.1 µs  75.4%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       9.0 µs   4.5%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       8.6 µs   4.3%      3
void at::native::unrolled_elementwise_kernel<at::native::...       6.9 µs   3.5%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.9 µs   2.5%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       3.1 µs   1.6%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.3%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.1 µs   1.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.9 µs   1.0%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.7 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   0.7%      1
void at::native::vectorized_elementwise_kernel<8, at::nat...       1.3 µs   0.7%      1
... 5 more
total                                                            197.8 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     190.0 µs  92.7%      2
_post_expert_norm_weighted_sum_kernel                              4.0 µs   1.9%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.4 µs   1.2%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.4 µs   1.2%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.1 µs   1.0%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.3 µs   0.7%      1
total                                                            204.9 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     116.9 µs  72.7%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       8.9 µs   5.5%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       6.6 µs   4.1%      3
void at::native::unrolled_elementwise_kernel<at::native::...       6.0 µs   3.7%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.4 µs   2.7%      1
_moe_align_small_numel_kernel                                      3.5 µs   2.2%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       2.9 µs   1.8%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.1 µs   1.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.8 µs   1.1%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.5 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   0.8%      1
void at::native::vectorized_elementwise_kernel<8, at::nat...       1.2 µs   0.8%      1
... 4 more
total                                                            160.8 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w4_gate_up                                                     1569.8 µs  62.2%      1
_w4_down                                                         937.0 µs  37.2%      1
_post_expert_norm_weighted_sum_kernel                              5.3 µs   0.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.3 µs   0.1%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.9 µs   0.1%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.1%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.1 µs   0.0%      1
total                                                           2522.1 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w4_gate_up                                                     1565.8 µs  62.4%      1
_w4_down                                                         925.0 µs  36.9%      1
_post_expert_norm_weighted_sum_kernel                              6.7 µs   0.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.2 µs   0.1%      1
void at::native::unrolled_elementwise_kernel<at::native::...       3.1 µs   0.1%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.8 µs   0.1%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   0.0%      1
total                                                           2507.9 µs
```

