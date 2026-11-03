Candidate: projection_fp32_splits_v1; private check 65ff350aa9884888
Independent one-case estimates; no speedup is claimed. Lower bounds are theoretical.
| Load | Layer | Time µs | Bound µs | Score gain for 1 µs saved | Score gain for 10% case speedup |
|---|---|---:|---:|---:|---:|
| decode 1 | qkv_gate | 8.90 | 7.87 | 0.493% | 0.398% |
| decode 1 | o | 5.31 | 3.50 | 0.493% | 0.236% |
| decode 1 | pooler | 13.21 | 13.99 | 0.493% | 0.595% |
| decode 1 | unpooler | 3.43 | 1.75 | 0.493% | 0.152% |
| decode 1 | router | 3.51 | 1.75 | 0.493% | 0.155% |
| decode 16 | qkv_gate | 9.75 | 7.93 | 0.357% | 0.316% |
| decode 16 | o | 5.44 | 3.56 | 0.357% | 0.176% |
| decode 16 | pooler | 14.00 | 14.06 | 0.357% | 0.456% |
| decode 16 | unpooler | 13.70 | 14.06 | 0.357% | 0.447% |
| decode 16 | router | 4.33 | 1.81 | 0.357% | 0.140% |
| decode 32 | qkv_gate | 14.42 | 8.00 | 0.308% | 0.404% |
| decode 32 | o | 5.84 | 3.62 | 0.308% | 0.162% |
| decode 32 | pooler | 14.72 | 14.14 | 0.308% | 0.413% |
| decode 32 | unpooler | 14.68 | 14.14 | 0.308% | 0.412% |
| decode 32 | router | 5.12 | 1.86 | 0.308% | 0.142% |
| verify 1 | qkv_gate | 9.30 | 7.90 | 0.462% | 0.390% |
| verify 1 | o | 5.50 | 3.53 | 0.462% | 0.229% |
| verify 1 | pooler | 13.61 | 14.02 | 0.462% | 0.574% |
| verify 1 | unpooler | 4.26 | 1.76 | 0.462% | 0.177% |
| verify 1 | router | 3.98 | 1.78 | 0.462% | 0.166% |
| verify 16 | qkv_gate | 15.18 | 8.42 | 0.255% | 0.354% |
| verify 16 | o | 6.45 | 3.99 | 0.255% | 0.149% |
| verify 16 | pooler | 19.45 | 14.64 | 0.255% | 0.455% |
| verify 16 | unpooler | 18.81 | 14.64 | 0.255% | 0.440% |
| verify 16 | router | 5.94 | 2.21 | 0.255% | 0.137% |
| verify 32 | qkv_gate | 16.94 | 9.77 | 0.209% | 0.323% |
| verify 32 | o | 8.51 | 4.48 | 0.209% | 0.161% |
| verify 32 | pooler | 24.40 | 17.37 | 0.209% | 0.468% |
| verify 32 | unpooler | 23.32 | 17.37 | 0.209% | 0.446% |
| verify 32 | router | 7.21 | 2.68 | 0.209% | 0.137% |

The 1 µs and 10% columns are mathematical sensitivities, not predictions. A theoretical bound can differ from measured timings because of assumptions, noise and cache behavior. Candidate eligibility still requires the precision audit in `research/compliance/2026-11-03-rule-audit.md`.
