# moe_fused_small_reduce_v1

Job: `7c48aa5ad79c4601`
Correctness: True; official speedup: 1.14626x; CI: [1.129063866288032, 1.1634515982561953]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 77.01 | 52.13 | 1.477x | 11.42 |
| decode 16 | experts (ep8) | 189.44 | 156.09 | 1.214x | 52.52 |
| decode 32 | experts (ep8) | 210.00 | 209.81 | 1.001x | 70.76 |
| verify 1 | experts (ep8) | 149.79 | 118.66 | 1.262x | 37.32 |
| verify 16 | experts (ep8) | 460.53 | 460.21 | 1.001x | 101.34 |
| verify 32 | experts (ep8) | 624.96 | 624.61 | 1.001x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.620 | 3.128 | 1.477x | 6.745x |
| decode 16 | 11.366 | 9.365 | 1.214x | 3.607x |
| decode 32 | 12.600 | 12.589 | 1.001x | 2.968x |
| verify 1 | 8.988 | 7.120 | 1.262x | 4.014x |
| verify 16 | 27.632 | 27.612 | 1.001x | 4.544x |
| verify 32 | 37.498 | 37.477 | 1.001x | 5.737x |

Recomputed geometric mean across loads: 1.146258x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      40.9 µs  81.7%      2
_post_expert_norm_weighted_sum_kernel                              3.0 µs   5.9%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.7 µs   3.4%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.6 µs   3.3%      1
_moe_align_small_numel_kernel                                      1.5 µs   3.0%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   2.8%      1
total                                                             50.0 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      97.4 µs  88.7%      2
_post_expert_norm_weighted_sum_kernel                              3.4 µs   3.1%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   2.4%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.0 µs   1.8%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.9 µs   1.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.2 µs   1.1%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   1.0%      1
total                                                            109.8 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     234.6 µs  93.4%      2
_post_expert_norm_weighted_sum_kernel                              5.8 µs   2.3%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.8 µs   1.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.4 µs   1.0%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.4 µs   1.0%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.9 µs   0.8%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   0.5%      1
total                                                            251.2 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      87.1 µs  87.8%      2
_post_expert_norm_weighted_sum_kernel                              3.5 µs   3.6%      1
_moe_align_small_numel_kernel                                      3.4 µs   3.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.2 µs   2.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.8 µs   1.8%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.2 µs   1.2%      1
total                                                             99.1 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     437.2 µs  95.7%      2
_post_expert_norm_weighted_sum_kernel                              5.2 µs   1.1%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.7 µs   0.8%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.3 µs   0.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.9 µs   0.6%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.6%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.7 µs   0.4%      1
total                                                            456.7 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     636.4 µs  96.5%      2
_post_expert_norm_weighted_sum_kernel                              6.0 µs   0.9%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.7 µs   0.9%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.4 µs   0.5%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.9 µs   0.4%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.9 µs   0.4%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       2.4 µs   0.4%      1
total                                                            659.7 µs
```

