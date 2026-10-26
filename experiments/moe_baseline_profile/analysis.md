# moe_baseline_profile

Job: `c761669d797940e1`
Correctness: True; official speedup: 1.00146x; CI: [0.9864342588263316, 1.0164779418362706]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 77.17 | 76.74 | 1.006x | 11.42 |
| decode 16 | experts (ep8) | 187.97 | 188.19 | 0.999x | 52.52 |
| decode 32 | experts (ep8) | 204.62 | 204.41 | 1.001x | 70.76 |
| verify 1 | experts (ep8) | 149.89 | 149.58 | 1.002x | 37.32 |
| verify 16 | experts (ep8) | 461.50 | 461.39 | 1.000x | 101.34 |
| verify 32 | experts (ep8) | 617.23 | 616.64 | 1.001x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.630 | 4.604 | 1.006x | 6.759x |
| decode 16 | 11.278 | 11.291 | 0.999x | 3.579x |
| decode 32 | 12.277 | 12.264 | 1.001x | 2.892x |
| verify 1 | 8.993 | 8.975 | 1.002x | 4.016x |
| verify 16 | 27.690 | 27.683 | 1.000x | 4.554x |
| verify 32 | 37.034 | 36.998 | 1.001x | 5.666x |

Recomputed geometric mean across loads: 1.001456x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      40.6 µs  54.5%      2
void at::native::unrolled_elementwise_kernel<at::native::...       5.5 µs   7.4%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       5.2 µs   7.1%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       4.7 µs   6.4%      3
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.3 µs   5.7%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       2.8 µs   3.7%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.5 µs   2.1%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.5 µs   2.0%      1
_moe_align_small_numel_kernel                                      1.3 µs   1.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.1 µs   1.5%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.1 µs   1.5%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.1 µs   1.4%      1
... 4 more
total                                                             74.4 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     129.0 µs  73.0%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       9.0 µs   5.1%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       8.6 µs   4.9%      3
void at::native::unrolled_elementwise_kernel<at::native::...       6.3 µs   3.5%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.9 µs   2.8%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       3.1 µs   1.8%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.1 µs   1.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.7 µs   1.0%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.7 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   0.8%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   0.7%      1
... 5 more
total                                                            176.7 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     169.1 µs  92.0%      2
_post_expert_norm_weighted_sum_kernel                              4.2 µs   2.3%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.3 µs   1.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.2 µs   1.2%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.0 µs   1.1%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.3 µs   0.7%      1
total                                                            183.8 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     115.7 µs  72.4%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       9.0 µs   5.6%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       6.6 µs   4.1%      3
void at::native::unrolled_elementwise_kernel<at::native::...       5.8 µs   3.6%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.4 µs   2.7%      1
_moe_align_small_numel_kernel                                      3.5 µs   2.2%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       3.5 µs   2.2%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.1 µs   1.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.6 µs   1.0%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.3 µs   0.8%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   0.8%      1
void at::native::vectorized_elementwise_kernel<8, at::nat...       1.3 µs   0.8%      1
... 4 more
total                                                            159.8 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     363.0 µs  95.0%      2
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.3%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.7 µs   1.0%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.1 µs   0.8%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.8 µs   0.7%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.7%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.8 µs   0.5%      1
total                                                            382.2 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     586.3 µs  96.4%      2
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.4 µs   0.9%      1
_post_expert_norm_weighted_sum_kernel                              5.2 µs   0.8%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.2 µs   0.5%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.9 µs   0.5%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.9 µs   0.5%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       2.4 µs   0.4%      1
total                                                            608.3 µs
```

