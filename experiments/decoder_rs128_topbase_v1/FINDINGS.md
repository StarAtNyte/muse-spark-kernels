# rs128 transfer on the exact scored decoder base

## Setup

Parent: exact `b72f02c7eb8745f5` source, SHA256
`9e7c8b73b0471d04f4f0cf451bea6ad372d7356e3103e7f57d7253df234aebb2`.
Candidate: rs128 QKV/unpooler transfer, SHA256
`90042cfd63543fccc49bb91b2eddc45553f8d3eb05cb4d578e728ff355ebc2ec`.
The candidate previously passed the official 13/13 correctness check
`28bf49b830714862`.

The H200 comparison used the official SGLang image, identical weights and
inputs, fresh seeds 4001–4003, 20 alternating cold-L2 graph replays per source,
and reference checks for both outputs. It covered verify-16/32, full/sliding.
Modal app: `ap-nQzMG0EsWGBXMRy7tr8sUz`.

## Paired result

Gain is parent latency divided by candidate latency; values above 1 favor the
candidate.

| Case | Seed 4001 | Seed 4002 | Seed 4003 | Interpretation |
|---|---:|---:|---:|---|
| Verify 16 sliding | 1.0110x | 1.0129x | 1.0104x | repeatable ~1.1% gain |
| Verify 16 full | 1.0103x | 1.0120x | 1.0101x | repeatable ~1.1% gain |
| Verify 32 sliding | 1.0002x | 0.9979x | 0.9985x | neutral within noise |
| Verify 32 full | 1.0035x | 1.0109x | 1.0051x | small ~0.65% gain |

The geometric mean over these 12 case/seed ratios is 1.0069x. This is a
selected-case screen, not a whole-decoder speedup estimate: the feature changes
only selected QKV/unpooler shapes, and the other eight official loads were not
included in this paired screen. It should therefore not be presented as a
0.69% whole-score gain. If the 12 official loads are equally weighted and the
other eight stay unchanged, applying this selected-case ratio to one third of
the score gives a rough projection of `1.0069^(4/12) = 1.0023x`, or +0.23%.
That is a hypothesis for the judge run, not an observed aggregate gain.

## Decision

Submitted as `1603d76ca92045e1` because verify-16 gains repeated in both layer
modes and the candidate already passed complete correctness. The scored run
passed 13/13 but regressed to 2.20383x (95% CI 2.17077–2.23689), versus the
2.26032x incumbent, so reject it and retain the incumbent. Its profile shows
the main regressions at decode-16 sliding (135.3 us vs 127.8 us) and decode-32
sliding (157.8 us vs 148.2 us); verify-1 full also regressed (123.1 us vs
114.4 us). The selected-case Modal screen did not predict this aggregate
outcome. Do not promote the rs128 candidate.

The largest remaining opportunity is still the integrated MoE path, which
occupies 52–57% of decode-16/32 sliding profile time and over 60% on some
verification cases. Next experiments should target that stage while keeping
the incumbent's load-specific choices intact.
