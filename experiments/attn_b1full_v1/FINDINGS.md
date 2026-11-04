# attn_b1full_v1: B=1 full-context attention (d1f / v1f)

Parent: experiments/attn_vfull_v1/submission_g16b.py (sha256 78fec1b5..., parent.py). Modal H200, judge image, cold-L2 graph
replays, paired, every graph reference-checked. Variants: make_var.py -> dev/modal/_b1full/.

## Phase stamps (us from first CTA start, median [max] over 30 reps; r1_stamp.log, r5_stamp.log)
d1f (252 CTAs, 2/SM, 5 tiles each, nc 8, coop): start 0.1 | page idx 0.96 | q group 1.73 | first tile 2.66 [3.49] |
last tile 6.02 [6.72] | merge start 6.62 [7.23] | reds issued 6.88 | arrive atomic returns 8.13 [8.58] | barrier 8.74 | end 8.8 [9.66].
v1f: idx 1.15 | q 1.76 | merge start 7.52 [9.04] | reds 8.16 | atomic 9.57 [10.5] | barrier 10.62 | end 11.49 [11.81].
- Stream (KV issue ~0.96 -> last tile 6.0) moves 20.5 MB at ~4.0 TB/s: near the practical rate for random 128 B rows.
  Little's law: 30 GB/s/SM x ~1.7 us loaded latency = ~51 KB/SM needed; 2 CTAs x 2 tiles x 16 KB = 64 KB in flight. Not the limit.
- The waste is the tail: the arrive atomic waits ~1.25 us (d1f) / 1.4 us (v1f) for the CTA's reds to be performed in L2, then
  the barrier waits for the slowest CTA (v1f merge-start spread 1.5 us), then the finalizers take ~0.9 us.
## Results (paired vs parent)
| variant | d1f | v1f | v1s | notes |
|---|---|---|---|---|
| bfd: bf16 v4.bf16x2 partials in the B=1 q1 coop spec build | 1.026-1.033 | 1.00 | - | atomic return 8.13 -> 7.84 |
| bfd + nc 4 / nc 16 | 0.98-0.99 / 1.005-1.024 | | | more copies or fewer: worse than nc 8 |
| v4r: q8 bf16 partials as v4.bf16x2 (quad transpose) instead of v2 | | 1.019-1.027 | 1.02-1.03 | v16s/v32s +0-1.7% |
| bv = bfd + v4r (candidate) | 1.024-1.042 | 1.016-1.029 | 1.005-1.011 | all 16 tests pass |
| bv + coop 0 (nc 16) | 0.91 | | | |
| bv + v1f nc 16 | | 0.89-0.90 | | rolled 4-copy sum loop |
| bv + flat q8 bf16 (v16f/v32f) | | | | 0.99-1.00: rejected |
| bv + no q1 half handoff (both halves red) | 0.99-1.00 | | | rejected |
Judge: bv check eeb6cd9dd01b440d 2.672x [2.632, 2.712] (parent g16b check 979f4c2f 2.652x): d1f 11.2 -> 10.4 us,
v1f 13.3 -> 12.8 us; other shapes within noise. Precision: d1f error 2.7e-3 -> 4.3e-3 (bound 0.087-0.108).
## Not done / next
- v1f imbalance (merge start median 7.65 vs max 9.1) and the ~1 us L2 release wait are what remain; 8 us d1f needs a merge
  that does not wait on L2 atomics (e.g. a DSMEM pre-reduce without cluster launch cost) or a finalize warmed in the i-cache.
Recheck (comment-only copy submission_bv_r2.py) 1c022f61098a41af: 2.665x [2.625, 2.714]; d1f 10.3, v1f 12.7 us. Both checks reproduce the
B=1 full gains (d1f -0.8..-0.9 us, v1f -0.5..-0.6 us). The board-level gain (+0.5..+0.8% vs the 2.652x parent check) is inside the CI.

## Stack (bv + bx16 plain q_len-1 bf16 partials, d16s at 16 splits): submission_stack.py
Modal r6_stack.log, all 12 shapes + 16 tests, paired vs g16b: d16s 1.018-1.051, d32s 0.994-1.03, d1f 1.019-1.056, v1f 1.018-1.025,
v1s 1.014-1.02, rest 0.99-1.01; all tests pass (sliding q1 errors up ~1.4x, far inside bounds).
No other hot 8-byte reds remain (the v2.bf16x2 left is in cluster_multi, unreachable from _plan).
Judge: 171dfdd2900f4adb 2.697x [2.657, 2.756]; comment-only copy d835372941034b88 2.733x [2.692, 2.774]. bv scored 2.682x.
