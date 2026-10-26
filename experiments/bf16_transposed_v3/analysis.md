# bf16_transposed_v3

Job: `6128e66277164905`
Correctness: True; official speedup: 0.84512x; CI: [0.8324464375591673, 0.8578001361650301]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 13.66 | 13.16 | 1.038x | 7.87 |
| decode 1 | o | 6.72 | 7.50 | 0.895x | 3.50 |
| decode 1 | pooler | 20.98 | 18.95 | 1.107x | 13.99 |
| decode 1 | unpooler | 6.88 | 6.70 | 1.026x | 1.75 |
| decode 1 | router | 10.16 | 6.64 | 1.531x | 1.75 |
| decode 16 | qkv_gate | 13.86 | 13.63 | 1.017x | 7.93 |
| decode 16 | o | 6.85 | 7.88 | 0.869x | 3.56 |
| decode 16 | pooler | 20.80 | 19.06 | 1.091x | 14.06 |
| decode 16 | unpooler | 18.56 | 18.90 | 0.982x | 14.06 |
| decode 16 | router | 14.78 | 6.84 | 2.163x | 1.81 |
| decode 32 | qkv_gate | 14.06 | 15.92 | 0.884x | 8.00 |
| decode 32 | o | 6.66 | 7.26 | 0.917x | 3.62 |
| decode 32 | pooler | 21.09 | 23.59 | 0.894x | 14.14 |
| decode 32 | unpooler | 18.70 | 22.29 | 0.839x | 14.14 |
| decode 32 | router | 15.36 | 7.36 | 2.088x | 1.86 |
| verify 1 | qkv_gate | 13.98 | 13.43 | 1.041x | 7.90 |
| verify 1 | o | 6.59 | 7.62 | 0.865x | 3.53 |
| verify 1 | pooler | 21.01 | 19.02 | 1.104x | 14.02 |
| verify 1 | unpooler | 6.98 | 6.82 | 1.023x | 1.76 |
| verify 1 | router | 26.43 | 7.31 | 3.614x | 1.78 |
| verify 16 | qkv_gate | 16.99 | 34.36 | 0.495x | 8.42 |
| verify 16 | o | 7.17 | 12.84 | 0.558x | 3.99 |
| verify 16 | pooler | 25.06 | 49.00 | 0.511x | 14.64 |
| verify 16 | unpooler | 19.63 | 41.87 | 0.469x | 14.64 |
| verify 16 | router | 16.40 | 9.79 | 1.676x | 2.21 |
| verify 32 | qkv_gate | 19.81 | 66.07 | 0.300x | 9.77 |
| verify 32 | o | 9.34 | 24.64 | 0.379x | 4.48 |
| verify 32 | pooler | 28.74 | 83.34 | 0.345x | 17.37 |
| verify 32 | unpooler | 24.29 | 82.98 | 0.293x | 17.37 |
| verify 32 | router | 17.62 | 12.96 | 1.360x | 2.68 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 3.621 | 3.283 | 1.103x | 2.024x |
| decode 16 | 4.641 | 4.111 | 1.129x | 1.807x |
| decode 32 | 4.704 | 4.738 | 0.993x | 1.816x |
| verify 1 | 4.650 | 3.361 | 1.383x | 2.587x |
| verify 16 | 5.285 | 9.167 | 0.577x | 1.942x |
| verify 32 | 6.187 | 16.739 | 0.370x | 1.931x |

Recomputed geometric mean across loads: 0.845123x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.6 µs  87.6%      1
_finish                                                            1.2 µs  12.4%      1
total                                                              9.8 µs
```

### decode 1: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            4.0 µs  81.0%      1
_finish                                                            0.9 µs  19.0%      1
total                                                              4.9 µs
```

### decode 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.2 µs  94.9%      1
_finish                                                            0.9 µs   5.1%      1
total                                                             18.1 µs
```

### decode 1: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.2 µs  67.1%      1
_finish                                                            1.6 µs  32.9%      1
total                                                              4.8 µs
```

### decode 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  72.2%      1
_finish                                                            1.4 µs  27.8%      1
total                                                              5.1 µs
```

### decode 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.3 µs  86.3%      1
_finish                                                            1.3 µs  13.7%      1
total                                                              9.6 µs
```

### decode 16: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            4.2 µs  78.4%      1
_finish                                                            1.2 µs  21.6%      1
total                                                              5.3 µs
```

### decode 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.8 µs  93.9%      1
_finish                                                            1.2 µs   6.1%      1
total                                                             19.0 µs
```

### decode 16: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           18.1 µs  93.9%      1
_finish                                                            1.2 µs   6.1%      1
total                                                             19.3 µs
```

### decode 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.4 µs  67.1%      1
_finish                                                            1.7 µs  32.9%      1
total                                                              5.1 µs
```

### decode 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           12.8 µs  92.0%      1
_finish                                                            1.1 µs   8.0%      1
total                                                             14.0 µs
```

### decode 32: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.2 µs 100.0%      1
total                                                              6.2 µs
```

### decode 32: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           22.5 µs  95.0%      1
_finish                                                            1.2 µs   5.0%      1
total                                                             23.6 µs
```

### decode 32: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           22.2 µs 100.0%      1
total                                                             22.2 µs
```

### decode 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.7 µs  68.1%      1
_finish                                                            1.7 µs  31.9%      1
total                                                              5.4 µs
```

### verify 1: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                            8.2 µs  86.4%      1
_finish                                                            1.3 µs  13.6%      1
total                                                              9.4 µs
```

### verify 1: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                            4.0 µs  79.1%      1
_finish                                                            1.1 µs  20.9%      1
total                                                              5.1 µs
```

### verify 1: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           17.7 µs  94.5%      1
_finish                                                            1.0 µs   5.5%      1
total                                                             18.7 µs
```

### verify 1: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.2 µs  66.0%      1
_finish                                                            1.7 µs  34.0%      1
total                                                              4.9 µs
```

### verify 1: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            3.6 µs  68.5%      1
_finish                                                            1.7 µs  31.5%      1
total                                                              5.3 µs
```

### verify 16: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           33.0 µs  95.7%      1
_finish                                                            1.5 µs   4.3%      1
total                                                             34.4 µs
```

### verify 16: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                           12.5 µs 100.0%      1
total                                                             12.5 µs
```

### verify 16: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           48.7 µs 100.0%      1
total                                                             48.7 µs
```

### verify 16: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           42.2 µs 100.0%      1
total                                                             42.2 µs
```

### verify 16: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                            6.4 µs  78.3%      1
_finish                                                            1.8 µs  21.7%      1
total                                                              8.1 µs
```

### verify 32: qkv_gate

```text
kernel                                                           GPU time  share  calls
_matmul                                                           66.8 µs 100.0%      1
total                                                             66.8 µs
```

### verify 32: o

```text
kernel                                                           GPU time  share  calls
_matmul                                                           23.9 µs 100.0%      1
total                                                             23.9 µs
```

### verify 32: pooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           83.9 µs 100.0%      1
total                                                             83.9 µs
```

### verify 32: unpooler

```text
kernel                                                           GPU time  share  calls
_matmul                                                           85.7 µs 100.0%      1
total                                                             85.7 µs
```

### verify 32: router

```text
kernel                                                           GPU time  share  calls
_matmul                                                           10.0 µs  87.5%      1
_finish                                                            1.4 µs  12.5%      1
total                                                             11.5 µs
```

