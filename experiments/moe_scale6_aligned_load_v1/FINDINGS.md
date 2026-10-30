# Aligned shared-memory loads for compact WGS scales

Each lane's six-byte scale group is read using two aligned 32-bit shared-memory loads, with shifts when its offset is 2 modulo 4. Exhaustive host-side bounds checking confirms that both words remain inside the 768-byte scale block for all 128 lane positions. The candidate preserves the full-byte fallback from `../moe_scale6_wgs_fallback_v1/`.

First, `probe.log` compared the candidate to the uncompressed parent. All six cases passed the reference check and were bit-identical; ratios ranged from 0.9864 to 1.0213. Then `paired.log` directly compared it with the three-16-bit-load compressed version using the same packed state and 20 alternating cold-L2 graph samples per shape. All six comparisons passed and were bit-identical:

| Tokens | Seed | Three-load µs | Aligned-load µs | Three-load / aligned-load |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 419 | 131.34 | 130.54 | 1.0061 |
| 32 | 7919 | 150.40 | 150.99 | 0.9961 |
| 128 | 419 | 191.01 | 190.83 | 1.0009 |
| 128 | 7919 | 187.57 | 188.61 | 0.9945 |
| 256 | 419 | 226.93 | 226.88 | 1.0002 |
| 256 | 7919 | 210.54 | 211.57 | 0.9952 |

No consistent benefit is established. Retain the previously scored compressed experiment for further investigation and the uncompressed public best as canonical; no official check was launched for this variant. These are component screens, not full judge validation.
