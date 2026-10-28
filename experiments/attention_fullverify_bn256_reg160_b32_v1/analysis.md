# attention_fullverify_bn256_reg160_b32_v1

Job: `4a99c00577844af7`
Correctness: True; official speedup: 1.78890x; CI: [1.7620682559685628, 1.8157353094498387]

| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |
|---|---|---:|---:|---:|---:|
| decode 1 | sliding | 20.30 | 7.85 | 2.587x | 0.11 |
| decode 1 | full | 30.61 | 15.80 | 1.938x | 4.27 |
| decode 16 | sliding | 27.01 | 12.33 | 2.191x | 1.65 |
| decode 16 | full | 200.06 | 97.09 | 2.061x | 63.18 |
| decode 32 | sliding | 31.04 | 14.61 | 2.124x | 3.29 |
| decode 32 | full | 383.15 | 183.94 | 2.083x | 126.35 |
| verify 1 | sliding | 22.74 | 8.66 | 2.624x | 0.12 |
| verify 1 | full | 35.30 | 22.50 | 1.568x | 4.28 |
| verify 16 | sliding | 29.47 | 17.23 | 1.711x | 1.86 |
| verify 16 | full | 204.22 | 155.14 | 1.316x | 63.39 |
| verify 32 | sliding | 34.58 | 21.07 | 1.641x | 3.72 |
| verify 32 | full | 389.41 | 393.76 | 0.989x | 126.78 |

| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |
|---|---:|---:|---:|---:|
| decode 1 | 1.424 | 0.614 | 2.320x | 19.394x |
| decode 16 | 4.443 | 2.120 | 2.096x | 4.089x |
| decode 32 | 7.558 | 3.615 | 2.091x | 3.478x |
| verify 1 | 1.611 | 0.759 | 2.123x | 21.693x |
| verify 16 | 4.623 | 3.275 | 1.412x | 4.203x |
| verify 32 | 7.821 | 7.270 | 1.076x | 3.555x |

Recomputed geometric mean across loads: 1.788902x.
Roofline bounds omit launch, merge, redundant traffic, and other costs.

## Official profiler reports

### decode 1: sliding

```text
kernel                                                           GPU time  share  calls
void fd2_partials<3>(unsigned char const*, unsigned char ...       4.3 µs  58.8%      1
_prepare_qkv                                                       1.5 µs  20.6%      1
_merge_norm_gate                                                   1.5 µs  20.6%      1
total                                                              7.3 µs
```

### decode 1: full

```text
kernel                                                           GPU time  share  calls
void fd2_partials<2>(unsigned char const*, unsigned char ...      10.3 µs  77.1%      1
_merge_norm_gate                                                   2.0 µs  15.0%      1
_prepare_qkv                                                       1.1 µs   7.9%      1
total                                                             13.4 µs
```

### decode 16: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       5.2 µs  61.9%      1
_merge_norm_gate                                                   1.7 µs  19.6%      1
_prepare_qkv                                                       1.6 µs  18.5%      1
total                                                              8.5 µs
```

### decode 16: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...      92.0 µs  94.9%      1
_merge_norm_gate                                                   3.1 µs   3.2%      1
_prepare_qkv                                                       1.9 µs   1.9%      1
total                                                             96.9 µs
```

### decode 32: sliding

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...       8.4 µs  70.5%      1
_merge_norm_gate                                                   1.8 µs  15.3%      1
_prepare_qkv                                                       1.7 µs  14.2%      1
total                                                             11.9 µs
```

### decode 32: full

```text
kernel                                                           GPU time  share  calls
void trans_decode<3>(unsigned char const*, unsigned char ...     180.0 µs  97.2%      1
_merge_norm_gate                                                   3.1 µs   1.7%      1
_prepare_qkv                                                       2.0 µs   1.1%      1
total                                                            185.2 µs
```

### verify 1: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                    7.2 µs 100.0%      1
total                                                              7.2 µs
```

### verify 1: full

```text
kernel                                                           GPU time  share  calls
wg5_partials(unsigned char const*, unsigned char const*, ...      16.3 µs  80.7%      1
_merge_norm_gate                                                   2.7 µs  13.3%      1
_prepare_qkv                                                       1.2 µs   6.0%      1
total                                                             20.2 µs
```

### verify 16: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   14.6 µs 100.0%      1
total                                                             14.6 µs
```

### verify 16: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               156.3 µs 100.0%      1
total                                                            156.3 µs
```

### verify 32: sliding

```text
kernel                                                           GPU time  share  calls
_os__fused_attn                                                   19.2 µs 100.0%      1
total                                                             19.2 µs
```

### verify 32: full

```text
kernel                                                           GPU time  share  calls
_os__g_full_verify                                               393.1 µs 100.0%      1
total                                                            393.1 µs
```

