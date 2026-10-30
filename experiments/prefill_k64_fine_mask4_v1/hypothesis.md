# K64 prefill MoE: omit the 160-row tile

The scored K64 implementation compiles fine remainder widths 160 and 192 in addition to 32, 64, 128 and 256. This experiment changes `MOE_FINE_MASK` from 6 to 4, removing only the 160-row tile. Counts from 129 through 160 instead use a 192-row tile. This trades up to 32 extra tensor rows for fewer compiled consumer variants and potentially less instruction-cache and scheduling pressure. It may lose because the large-token MoE is already near tensor-pipe saturation; the direction needs measurement. `candidate.py` differs from the promoted source only by that constant.

`bench.py` uses paired cold-L2 full-layer graph timing with pinned attention and an SGLang import stub. It is a screen, not an official judge or full correctness test.

One H200 screen (`bench.log`, 8 alternating samples per shape, one input seed):

| Shape | Parent µs | Mask 4 µs | Parent / candidate |
| --- | ---: | ---: | ---: |
| batch 1, q753, sliding, context 80000 | 512.6 | 511.3 | 1.0024 |
| batch 4, q753, full, context 80000 | 2999.5 | 3005.3 | 0.9981 |
| batch 1, q16384, full, context 0 | 8774.3 | 8771.2 | 1.0004 |
| batch 1, q16384, full, context 65536 | 15575.3 | 15657.0 | 0.9948 |

Attention tuning failed under the import stub and fixed choices were used. No meaningful speedup is established, and the candidate did not receive the official correctness suite. Retain the scored parent and do not submit this change. A larger sample could change the estimate; the observed shape-level differences are small.
