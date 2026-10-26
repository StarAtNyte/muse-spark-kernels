# moe_hybrid_v5

Job: `c0b80be362af4fcf`
Correctness: True; official speedup: 1.26515x; CI: [1.246173534963258, 1.2841280588707682]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 77.23 | 52.61 | 1.468x | 11.42 |
| decode 16 | experts (ep8) | 189.10 | 155.42 | 1.217x | 52.52 |
| decode 32 | experts (ep8) | 200.83 | 200.98 | 0.999x | 70.76 |
| verify 1 | experts (ep8) | 150.34 | 119.01 | 1.263x | 37.32 |
| verify 16 | experts (ep8) | 460.24 | 337.09 | 1.365x | 101.34 |
| verify 32 | experts (ep8) | 620.88 | 466.10 | 1.332x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.634 | 3.156 | 1.468x | 6.765x |
| decode 16 | 11.346 | 9.325 | 1.217x | 3.601x |
| decode 32 | 12.050 | 12.059 | 0.999x | 2.838x |
| verify 1 | 9.020 | 7.141 | 1.263x | 4.028x |
| verify 16 | 27.614 | 20.225 | 1.365x | 4.541x |
| verify 32 | 37.253 | 27.966 | 1.332x | 5.699x |

Recomputed geometric mean across loads: 1.265151x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      64.5 µs  86.1%      2
_post_expert_norm_weighted_sum_kernel                              4.5 µs   6.1%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       1.8 µs   2.4%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.6 µs   2.2%      1
_moe_align_small_numel_kernel                                      1.2 µs   1.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.2 µs   1.5%      1
total                                                             74.8 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     107.4 µs  89.4%      2
_post_expert_norm_weighted_sum_kernel                              3.3 µs   2.8%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   2.3%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.3 µs   1.9%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.0 µs   1.7%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.3 µs   1.1%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   1.0%      1
total                                                            120.2 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     191.1 µs  92.9%      2
_post_expert_norm_weighted_sum_kernel                              4.3 µs   2.1%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.8 µs   1.4%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.4 µs   1.2%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       2.3 µs   1.1%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.7 µs   0.8%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.2 µs   0.6%      1
total                                                            205.7 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...      86.7 µs  87.7%      2
_post_expert_norm_weighted_sum_kernel                              3.7 µs   3.7%      1
_moe_align_small_numel_kernel                                      3.4 µs   3.4%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       2.0 µs   2.0%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       1.9 µs   1.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       1.2 µs   1.2%      1
total                                                             98.8 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     274.6 µs  93.5%      2
_post_expert_norm_weighted_sum_kernel                              4.9 µs   1.7%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       3.7 µs   1.3%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.3 µs   1.1%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.7 µs   0.9%      1
void at::native::unrolled_elementwise_kernel<at::native::...       2.5 µs   0.9%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       1.9 µs   0.6%      1
total                                                            293.6 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56...     552.9 µs  95.8%      2
_post_expert_norm_weighted_sum_kernel                              6.5 µs   1.1%      1
void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A...       6.0 µs   1.0%      1
void at::native::index_elementwise_kernel<128, 4, at::nat...       3.4 µs   0.6%      1
void at::native::unrolled_elementwise_kernel<at::native::...       3.1 µs   0.5%      1
void moe_align_block_size_kernel<int>(int const*, int*, i...       2.9 µs   0.5%      1
void count_and_sort_expert_tokens_kernel<int>(int const*,...       2.4 µs   0.4%      1
total                                                            577.2 µs
```

