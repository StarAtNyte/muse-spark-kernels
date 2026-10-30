# Six-bit WGS scale experiment

This experimental source packs each group of eight E4M3 scale bytes into six bytes, reducing each WGS weight stage from 36 KiB to 35 KiB. It decodes pairs in the WGMMA consumer and leaves all other paths unchanged. This first candidate rejects scales outside `0x50..0x8f` and is not a valid general submission; the fallback variant is in `../moe_scale6_wgs_fallback_v1/`.

On H200 app `ap-CJ8u8XIG5DLP2TripCVSqD`, both outputs passed the reference check and were bit-identical for the six tested cases. Paired cold-L2 graph timing, median of 20 alternating samples, one weight seed and two routing/input seeds:

| Tokens | Seed | Parent µs | Compact µs | Parent / compact |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 419 | 132.72 | 130.14 | 1.0198 |
| 32 | 7919 | 152.93 | 150.34 | 1.0172 |
| 128 | 419 | 195.54 | 191.23 | 1.0225 |
| 128 | 7919 | 190.80 | 187.66 | 1.0167 |
| 256 | 419 | 228.93 | 226.48 | 1.0108 |
| 256 | 7919 | 209.52 | 211.41 | 0.9911 |

These are component timings, not a judge score. The experiment shares the remaining prepared state and uses an independently generated parent WGS copy. Full raw output is `probe.log`; full correctness, arbitrary scales, and wider conditions remain to be verified.
