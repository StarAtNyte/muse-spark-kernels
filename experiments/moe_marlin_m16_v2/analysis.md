# moe_marlin_m16_v2

Job: `8bf6b00f456e4952`
Correctness: True; official speedup: 1.10929x; CI: [1.092646572593738, 1.1259251484087756]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 80.21 | 79.81 | 1.005x | 11.42 |
| decode 16 | experts (ep8) | 177.79 | 177.80 | 1.000x | 52.52 |
| decode 32 | experts (ep8) | 215.15 | 215.21 | 1.000x | 70.76 |
| verify 1 | experts (ep8) | 142.61 | 142.35 | 1.002x | 37.32 |
| verify 16 | experts (ep8) | 462.22 | 334.35 | 1.382x | 101.34 |
| verify 32 | experts (ep8) | 622.98 | 465.24 | 1.339x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.812 | 4.789 | 1.005x | 7.026x |
| decode 16 | 10.668 | 10.668 | 1.000x | 3.385x |
| decode 32 | 12.909 | 12.913 | 1.000x | 3.041x |
| verify 1 | 8.556 | 8.541 | 1.002x | 3.821x |
| verify 16 | 27.733 | 20.061 | 1.382x | 4.561x |
| verify 32 | 37.379 | 27.914 | 1.339x | 5.718x |

Recomputed geometric mean across loads: 1.109286x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      40.8 µs  54.6%      2
void at::native::unrolled_elementwise_kernel<at::native::...       5.6 µs   7.5%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       5.2 µs   7.0%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       4.8 µs   6.5%      3
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.0 µs   5.4%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       2.7 µs   3.6%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.9 µs   2.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.4 µs   1.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   1.8%      1
_moe_align_small_numel_kernel                                      1.3 µs   1.7%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.1 µs   1.5%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.0 µs   1.3%      1
... 4 more
total                                                             74.7 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     150.5 µs  75.8%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       9.0 µs   4.5%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       8.6 µs   4.3%      3
void at::native::unrolled_elementwise_kernel<at::native::...       6.0 µs   3.0%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.5 µs   2.3%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       3.2 µs   1.6%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.4%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.2 µs   1.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.9 µs   1.0%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.7 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   0.7%      1
void at::native::vectorized_elementwise_kernel<8, at::nat...       1.4 µs   0.7%      1
... 5 more
total                                                            198.6 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     201.6 µs  93.3%      2
_post_expert_norm_weighted_sum_kernel                              4.1 µs   1.9%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.4 µs   1.1%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.4 µs   1.1%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.8 µs   0.8%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.3 µs   0.6%      1
total                                                            216.1 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      96.5 µs  68.5%      2
void at::native::reduce_kernel<512, 1, at::native::Reduce...       8.9 µs   6.3%      1
void at::native::elementwise_kernel<128, 2, at::native::g...       6.7 µs   4.8%      3
void at::native::unrolled_elementwise_kernel<at::native::...       5.8 µs   4.1%      2
void at::native::reduce_kernel<128, 4, at::native::Reduce...       4.1 µs   2.9%      1
_moe_align_small_numel_kernel                                      3.4 µs   2.4%      1
void at::native::elementwise_kernel<128, 4, at::native::g...       3.2 µs   2.3%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.7 µs   1.9%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.0 µs   1.4%      1
void at::native::vectorized_elementwise_kernel<4, at::nat...       1.3 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   0.9%      1
void at::native::vectorized_elementwise_kernel<8, at::nat...       1.2 µs   0.9%      1
... 4 more
total                                                            140.8 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     294.4 µs  93.9%      2
_post_expert_norm_weighted_sum_kernel                              4.8 µs   1.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.8 µs   1.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.4 µs   1.1%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.8 µs   0.9%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.9%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.7 µs   0.6%      1
total                                                            313.6 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     487.2 µs  95.2%      2
_post_expert_norm_weighted_sum_kernel                              6.7 µs   1.3%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.8 µs   1.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.4 µs   0.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       3.1 µs   0.6%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       3.0 µs   0.6%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       2.6 µs   0.5%      1
total                                                            511.7 µs
```

