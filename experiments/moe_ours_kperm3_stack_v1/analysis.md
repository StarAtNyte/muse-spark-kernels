# moe_ours_kperm3_stack_v1

Job: `2ecdb5bf891d40e4`
Correctness: True; official speedup: 2.39112x; CI: [2.3552508505567338, 2.426984379000086]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 76.61 | 24.47 | 3.130x | 11.42 |
| decode 16 | experts (ep8) | 185.36 | 91.00 | 2.037x | 52.52 |
| decode 32 | experts (ep8) | 202.24 | 121.40 | 1.666x | 70.76 |
| verify 1 | experts (ep8) | 149.34 | 67.23 | 2.221x | 37.32 |
| verify 16 | experts (ep8) | 472.10 | 176.14 | 2.680x | 101.34 |
| verify 32 | experts (ep8) | 615.38 | 208.23 | 2.955x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.596 | 1.468 | 3.130x | 6.710x |
| decode 16 | 11.122 | 5.460 | 2.037x | 3.529x |
| decode 32 | 12.134 | 7.284 | 1.666x | 2.858x |
| verify 1 | 8.961 | 4.034 | 2.221x | 4.002x |
| verify 16 | 28.326 | 10.568 | 2.680x | 4.658x |
| verify 32 | 36.923 | 12.494 | 2.955x | 5.649x |

Recomputed geometric mean across loads: 2.391118x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           29.3 µs 100.0%      1
total                                                             29.3 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                          100.8 µs 100.0%      1
total                                                            100.8 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                          112.9 µs 100.0%      1
total                                                            112.9 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           42.8 µs 100.0%      1
total                                                             42.8 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           148.4 µs 100.0%      1
total                                                            148.4 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           193.7 µs 100.0%      1
total                                                            193.7 µs
```

