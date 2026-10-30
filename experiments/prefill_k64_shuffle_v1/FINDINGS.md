# K64 prefill gather with warp shuffles

This variant retains the K64 weight and gated intermediate permutation from `experiments/prefill_k64_fused_v1`, but replaces its 8 KiB shared-memory exchange and extra block barrier with a 4-lane register transpose. A lane owns eight BF16 pairs. For output pair `i`, the desired input resides in lane `i & 3`, pair `2*lane + (i >> 2)`; two butterfly shuffles exchange the two lane bits with the two register-index bits. An exhaustive CPU simulation checked all 32 pair positions in a 4-lane group.

The persistent MoE H200 screen compared the candidate against the current parent with one expert per token. Relative L2 differences were 0.00017254, 0.00016041 and 0.00013829 at 32, 96 and 753 tokens, matching the prior shared-memory variant. This is a narrow screen; special scales, other routing patterns, the full official suite and performance still require assessment. See `probe.log`.

H200 paired whole-layer timing (`bench.log`, app `ap-OPNBZc9WXJnYTJUIk7I3ud`, one seed, median of eight alternating cold-L2 graph replays):

| Shape | Parent µs | Shuffle candidate µs | Parent / candidate |
| --- | ---: | ---: | ---: |
| batch 1, q 753, sliding, context 80000 | 524.3 | 520.6 | 1.0071 |
| batch 4, q 753, full, context 80000 | 3055.9 | 3052.8 | 1.0010 |
| batch 1, q 16384, full, context 0 | 9251.3 | 9152.6 | 1.0108 |
| batch 1, q 16384, full, context 65536 | 15925.8 | 15979.9 | 0.9966 |

The benchmark uses pinned attention with an SGLang import stub; its attention tuner reported a stub-related error. These small mixed differences are not an official score or a reliable proof of overall gain. Official check `7d7b5055714b4239` was launched to test the full correctness suite and score.
