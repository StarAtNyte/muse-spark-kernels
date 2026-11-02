# Score-weighted kernel priorities (2026-11-02)

Generated from the latest available official scored profiles with
`tools/prioritize.py`. A 20% reduction below is a sensitivity calculation for
one timed shape inside its own problem's geometric-mean score. It is not a
predicted kernel gain, and the reference-model floor is optimistic rather than
a guaranteed physical limit.

Sources: projections `e5cae4f990be4df2`, attention `7d250883e6b94c50`, MoE
`43e6e35388774f73`, whole decoder `01bd05039098464e`, and prefill
`f3b10ea957e04ea6`. The live board snapshot and #50 gaps are in
`standings/2026-11-02.json`.

## Highest-leverage cases by kernel

### BF16 projections

| Load / layer | Time | Share of load | Score gain for 20% cut | Floor-model gain |
|---|---:|---:|---:|---:|
| Decode 1 / pooler | 13.05 us | 38.9% | 1.36% | 0.00% |
| Verify 1 / pooler | 13.62 us | 37.4% | 1.31% | 0.00% |
| Verify 32 / pooler | 24.04 us | 30.0% | 1.04% | 1.46% |

The singleton pooler GEMV is the first candidate target; broad rewrites of
small output projections have lower score leverage. Its time is already below
the reference's BF16 byte-floor estimate because the incumbent compresses
weights losslessly. A further win must beat the incumbent's compressed-byte
traffic and kernel overhead, not merely a plain BF16 streaming floor.

### Gated attention decode

| Load / layer | Time | Share of load | Score gain for 20% cut | Floor-model gain |
|---|---:|---:|---:|---:|
| Decode 32 / full | 165.94 us | 85.2% | 3.16% | 3.86% |
| Verify 32 / full | 184.86 us | 81.3% | 3.00% | 5.04% |
| Decode 16 / full | 86.80 us | 77.6% | 2.85% | 4.04% |
| Verify 16 / full | 98.98 us | 75.3% | 2.76% | 5.40% |

These full-context cases dominate their respective loads. Traffic accounting
and NCU show page-size-1 random KV reads already close to the measured H200
gather comparator; prioritize genuine reuse or fewer instructions/syncs on the
critical path, not generic TMA staging or CTA-count-only sweeps.

### Routed MoE W4A16

Each of the six scored loads has one expert-kernel case and contributes equally
to the geometric mean. A 20% reduction in any one case raises the problem
score by about 3.79%:

| Load | Time |
|---|---:|
| Decode 1 | 25.16 us |
| Decode 16 | 91.46 us |
| Decode 32 | 127.40 us |
| Verify 1 | 65.19 us |
| Verify 16 | 180.31 us |
| Verify 32 | 211.24 us |

SC6 WGS v9 (`bd7ee2f9d89247c8`) passed correctness at 2.36004x
(95% CI 2.32464–2.39544x). Scored run `1fb6b4f749ec44bb` confirmed the
`-DOKC_SC6W=1` route was active and scored 2.35842x (95% CI 2.32304–2.39379x),
still below the 2.38203x incumbent point estimate. Keep the incumbent; the
interval overlaps and the evidence does not show an aggregate gain. Keep all
shape-specific routes in view: no single batch size dominates the MoE score.

### Whole decoder layer

| Load / layer | Time | Share of load | Score gain for 20% cut | Floor-model gain |
|---|---:|---:|---:|---:|
| Verify 1 / sliding | 106.76 us | 73.6% | 2.69% | 6.29% |
| Decode 1 / sliding | 64.96 us | 72.5% | 2.65% | 5.59% |
| Verify 16 / sliding | 236.37 us | 67.9% | 2.46% | 6.52% |
| Verify 32 / sliding | 306.51 us | 65.2% | 2.36% | 7.33% |
| Decode 16 / sliding | 123.75 us | 63.2% | 2.28% | 4.69% |
| Decode 32 / sliding | 145.63 us | 58.4% | 2.09% | 4.80% |

The integrated decoder is the overall leaderboard score. Optimize and validate
these sliding cases in the complete layer: component-only attention or MoE
gains do not prove whole-layer improvement when PDL and stream overlap shift
the critical path. The rs128 QKV/unpooler transfer showed paired H200 gains on
selected verify cases, but its scored submission `1603d76ca92045e1` regressed
to 2.20383x (95% CI 2.17077–2.23689) from the 2.26032x incumbent. Reject it.
The official profile exposes decode-16/32 sliding regressions. Target the
integrated MoE path next; it consumes 52–57% of those loads.

### Prefill layer

| Load / context | Time | Share of load | Score gain for 20% cut | Floor-model gain |
|---|---:|---:|---:|---:|
| First turn / sliding | 8,765.88 us | 70.2% | 5.17% | 9.99% |
| Turn 1 / sliding | 517.97 us | 66.1% | 4.84% | 14.45% |
| Turn 4 / sliding | 1,759.35 us | 65.8% | 4.81% | 11.07% |

Long sliding prefill and short sliding contexts dominate. Existing K64 and
cuBLASLt transfers did not improve the official scored aggregate, so new
experiments should target the measured full-layer critical path and must
survive the official run before promotion. Prefill is a separate board.

## Validation commands

```bash
python tools/prioritize.py --problem muse_bf16_proj --status /path/to/status.json
python tools/prioritize.py --problem muse_attention_decode --status /path/to/status.json
python tools/prioritize.py --problem muse_moe_w4a16 --status /path/to/status.json
python tools/prioritize.py --problem muse_decode_layer --status /path/to/status.json
python tools/prioritize.py --problem muse_prefill_layer --status /path/to/status.json
```
