# moe_singleton_v7

Job: `52f4f669d02647e1`
Correctness: True; official speedup: 1.07960x; CI: [1.0634105796290647, 1.0957987191101528]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 79.94 | 144.95 | 0.551x | 11.42 |
| decode 16 | experts (ep8) | 190.83 | 157.35 | 1.213x | 52.52 |
| decode 32 | experts (ep8) | 216.02 | 216.05 | 1.000x | 70.76 |
| verify 1 | experts (ep8) | 145.87 | 115.30 | 1.265x | 37.32 |
| verify 16 | experts (ep8) | 458.16 | 332.71 | 1.377x | 101.34 |
| verify 32 | experts (ep8) | 630.21 | 463.70 | 1.359x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.796 | 8.697 | 0.551x | 7.002x |
| decode 16 | 11.450 | 9.441 | 1.213x | 3.634x |
| decode 32 | 12.961 | 12.963 | 1.000x | 3.053x |
| verify 1 | 8.752 | 6.918 | 1.265x | 3.909x |
| verify 16 | 27.490 | 19.963 | 1.377x | 4.521x |
| verify 32 | 37.812 | 27.822 | 1.359x | 5.785x |

Recomputed geometric mean across loads: 1.079605x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_single_gate_up                                                   62.3 µs  54.9%      1
_single_down                                                      48.3 µs  42.5%      1
_post_expert_norm_weighted_sum_kernel                              2.9 µs   2.5%      1
total                                                            113.5 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     127.8 µs  91.0%      2
_post_expert_norm_weighted_sum_kernel                              3.2 µs   2.3%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.9%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.2 µs   1.6%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.0 µs   1.4%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.4 µs   1.0%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.3 µs   0.9%      1
total                                                            140.5 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     134.1 µs  90.2%      2
_post_expert_norm_weighted_sum_kernel                              3.6 µs   2.4%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.7 µs   1.8%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   1.8%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.5 µs   1.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.7 µs   1.2%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.3 µs   0.9%      1
total                                                            148.7 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      96.4 µs  89.0%      2
_moe_align_small_numel_kernel                                      3.5 µs   3.2%      1
_post_expert_norm_weighted_sum_kernel                              3.2 µs   2.9%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.0 µs   1.9%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.0 µs   1.8%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   1.2%      1
total                                                            108.4 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     314.4 µs  94.2%      2
_post_expert_norm_weighted_sum_kernel                              5.1 µs   1.5%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.9 µs   1.2%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.4 µs   1.0%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.8%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.5 µs   0.8%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.8 µs   0.5%      1
total                                                            333.9 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     496.1 µs  95.5%      2
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       5.8 µs   1.1%      1
_post_expert_norm_weighted_sum_kernel                              5.8 µs   1.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.3 µs   0.6%      1
void at::native::unrolled_elementwise_kernel<at::native::...       3.0 µs   0.6%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.8 µs   0.5%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       2.5 µs   0.5%      1
total                                                            519.5 µs
```

