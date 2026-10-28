# attention_slide_fa3_nosplit

Job: `0707c94f18a34994`
Correctness: True; official speedup: 1.57425x; CI: [1.5506325959922818, 1.5978599847027066]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.06 | 7.51 | 2.670x | 0.11 |
| decode 1 | full | 31.22 | 15.57 | 2.005x | 4.27 |
| decode 16 | sliding | 26.99 | 13.56 | 1.990x | 1.65 |
| decode 16 | full | 201.28 | 121.70 | 1.654x | 63.18 |
| decode 32 | sliding | 31.44 | 17.17 | 1.831x | 3.29 |
| decode 32 | full | 385.44 | 208.96 | 1.845x | 126.35 |
| verify 1 | sliding | 22.37 | 13.23 | 1.691x | 0.12 |
| verify 1 | full | 36.30 | 26.94 | 1.348x | 4.28 |
| verify 16 | sliding | 29.09 | 29.77 | 0.977x | 1.86 |
| verify 16 | full | 205.28 | 173.90 | 1.180x | 63.39 |
| verify 32 | sliding | 34.30 | 30.39 | 1.129x | 3.72 |
| verify 32 | full | 390.62 | 336.36 | 1.161x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.422 | 0.595 | 2.392x | 19.376x |
| decode 16 | 4.462 | 2.571 | 1.736x | 4.106x |
| decode 32 | 7.613 | 4.133 | 1.842x | 3.503x |
| verify 1 | 1.610 | 1.040 | 1.549x | 21.683x |
| verify 16 | 4.623 | 4.152 | 1.113x | 4.203x |
| verify 32 | 7.828 | 6.780 | 1.155x | 3.558x |

Recomputed geometric mean across loads: 1.574246x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.4 µs  57.2%      1
_prepare_qkv                                                       1.7 µs  22.7%      1
_merge_norm_gate                                                   1.5 µs  20.2%      1
total                                                              7.6 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.4 µs  76.4%      1
_merge_norm_gate                                                   2.1 µs  15.6%      1
_prepare_qkv                                                       1.1 µs   8.0%      1
total                                                             13.6 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       7.1 µs  71.0%      1
_prepare_qkv                                                       1.5 µs  14.6%      1
_merge_norm_gate                                                   1.4 µs  14.3%      1
total                                                             10.0 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     115.1 µs  94.9%      1
_merge_norm_gate                                                   4.2 µs   3.5%      1
_prepare_qkv                                                       2.0 µs   1.6%      1
total                                                            121.3 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...      11.2 µs  77.0%      1
_prepare_qkv                                                       1.7 µs  11.9%      1
_merge_norm_gate                                                   1.6 µs  11.0%      1
total                                                             14.5 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void fd3_partials<2>(unsigned char const*, unsigned char ...     202.6 µs  97.0%      1
_merge_norm_gate                                                   4.3 µs   2.0%      1
_prepare_qkv                                                       2.0 µs   1.0%      1
total                                                            208.8 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                  6.1 µs  67.5%      1
_merge_norm_gate                                                   1.6 µs  17.3%      1
_prepare_qkv                                                       1.4 µs  15.2%      1
total                                                              9.1 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
_sliding_partials                                                 18.7 µs  82.9%      1
_merge_norm_gate                                                   2.7 µs  11.9%      1
_prepare_qkv                                                       1.2 µs   5.2%      1
total                                                             22.6 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      19.4 µs  80.4%      1
_prepare_qkv                                                       1.9 µs   8.0%      1
void flash::prepare_varlen_num_blocks_kernel<1, false>(in...       1.5 µs   6.4%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 8, 8, t...       1.3 µs   5.3%      1
total                                                             24.1 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     167.4 µs  96.0%      1
_merge_norm_gate                                                   4.8 µs   2.8%      1
_prepare_qkv                                                       2.1 µs   1.2%      1
total                                                            174.3 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...      20.2 µs  78.5%      1
_prepare_qkv                                                       2.6 µs  10.1%      1
void flash::prepare_varlen_num_blocks_kernel<2, false>(in...       1.5 µs   5.7%      1
void sglang::norm_gate_kernel<__nv_bfloat16, 64l, 16, 4, ...       1.5 µs   5.7%      1
total                                                             25.7 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...     322.0 µs  95.8%      1
_merge_norm_gate                                                  11.7 µs   3.5%      1
_prepare_qkv                                                       2.3 µs   0.7%      1
total                                                            336.0 µs
```

