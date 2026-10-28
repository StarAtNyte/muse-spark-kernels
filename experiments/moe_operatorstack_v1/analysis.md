# moe_operatorstack_v1

Job: `416e25689b8e4426`
Correctness: True; official speedup: 1.84636x; CI: [1.8186620965364082, 1.874052820288786]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 79.09 | 36.18 | 2.186x | 11.42 |
| decode 16 | experts (ep8) | 186.61 | 111.08 | 1.680x | 52.52 |
| decode 32 | experts (ep8) | 205.26 | 141.32 | 1.452x | 70.76 |
| verify 1 | experts (ep8) | 142.24 | 85.04 | 1.673x | 37.32 |
| verify 16 | experts (ep8) | 461.33 | 217.32 | 2.123x | 101.34 |
| verify 32 | experts (ep8) | 619.30 | 296.06 | 2.092x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.745 | 2.171 | 2.186x | 6.928x |
| decode 16 | 11.196 | 6.665 | 1.680x | 3.553x |
| decode 32 | 12.316 | 8.479 | 1.452x | 2.901x |
| verify 1 | 8.534 | 5.103 | 1.673x | 3.811x |
| verify 16 | 27.680 | 13.039 | 2.123x | 4.552x |
| verify 32 | 37.158 | 17.764 | 2.092x | 5.685x |

Recomputed geometric mean across loads: 1.846357x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w13_silu                                                         18.1 µs  48.5%      1
_w2_ks                                                            12.1 µs  32.4%      1
_wsum_ks                                                           5.5 µs  14.8%      1
_d1_reduce                                                         1.6 µs   4.4%      1
total                                                             37.4 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w2                                                               94.4 µs  94.0%      2
_post_expert_norm_weighted_sum_kernel                              3.3 µs   3.3%      1
_route                                                             2.7 µs   2.7%      1
total                                                            100.4 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w2                                                              132.1 µs  94.9%      2
_post_expert_norm_weighted_sum_kernel                              4.5 µs   3.2%      1
_route                                                             2.6 µs   1.9%      1
total                                                            139.2 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w13_silu                                                         48.4 µs  55.8%      1
_w2                                                               31.0 µs  35.8%      1
_post_expert_norm_weighted_sum_kernel                              5.0 µs   5.7%      1
_route                                                             2.3 µs   2.7%      1
total                                                             86.6 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w13_mix                                                         151.0 µs  63.8%      1
_w2_mix                                                           74.5 µs  31.5%      1
_route2                                                            6.4 µs   2.7%      1
_post_expert_norm_weighted_sum_kernel                              4.9 µs   2.1%      1
total                                                            236.7 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
_w13_mix                                                         204.9 µs  59.5%      1
_w2_mix                                                          122.7 µs  35.6%      1
_route2                                                           10.8 µs   3.1%      1
_post_expert_norm_weighted_sum_kernel                              5.8 µs   1.7%      1
total                                                            344.2 µs
```

