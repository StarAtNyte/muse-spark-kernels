# scored_moe_ours_kperm3_v1

Job: `190d8914bacb4aa2`
Correctness: True; official speedup: 2.38203x; CI: [2.3462966674247463, 2.417757479630576]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | experts (ep8) | 79.87 | 25.53 | 3.129x | 11.42 |
| decode 16 | experts (ep8) | 188.02 | 94.68 | 1.986x | 52.52 |
| decode 32 | experts (ep8) | 205.79 | 123.44 | 1.667x | 70.76 |
| verify 1 | experts (ep8) | 150.93 | 66.87 | 2.257x | 37.32 |
| verify 16 | experts (ep8) | 457.76 | 173.02 | 2.646x | 101.34 |
| verify 32 | experts (ep8) | 619.46 | 209.74 | 2.953x | 108.94 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 4.792 | 1.532 | 3.129x | 6.996x |
| decode 16 | 11.281 | 5.681 | 1.986x | 3.580x |
| decode 32 | 12.348 | 7.407 | 1.667x | 2.908x |
| verify 1 | 9.056 | 4.012 | 2.257x | 4.044x |
| verify 16 | 27.466 | 10.381 | 2.646x | 4.517x |
| verify 32 | 37.167 | 12.584 | 2.953x | 5.686x |

Recomputed geometric mean across loads: 2.382027x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           22.7 µs 100.0%      1
total                                                             22.7 µs
```

### decode 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           94.4 µs 100.0%      1
total                                                             94.4 µs
```

### decode 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                          137.9 µs 100.0%      1
total                                                            137.9 µs
```

### verify 1: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<false>(okc::Params)                           55.7 µs 100.0%      1
total                                                             55.7 µs
```

### verify 16: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           146.7 µs 100.0%      1
total                                                            146.7 µs
```

### verify 32: experts (ep8)

```text
kernel                                                           GPU time  share  calls
void okc::moe_fused<true>(okc::Params)                           212.6 µs 100.0%      1
total                                                            212.6 µs
```

