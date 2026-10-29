# Single-token split-K MoE (sk1) — correct, rejected (slower)

Idea: remove the gate/up -> down cross-CTA dependency at bs 1 by splitting the down projection's K into 16-wide
chunks. Unit (expert, chunk) = two 36 KB gate/up blocks of the unit-order copy + one 36 KB chunk of a new lossless
chunk-major down copy (`_sk1_pack_down`, verified exact by `dev/moe_splitk/test_pack.py`). Partial outputs
accumulate per CTA, flush with `red.global.add.v4.f32`, one grid barrier, then post-expert norm + weighted sum.
Source: `dev/moe_splitk/sk1.cu`, `dev/moe_splitk/build.py`; drivers `dev/modal/drv_moe_sk1.py`,
`dev/modal/drv_moe_sk1_prof.py`; logs here.

Correctness: 40/40 eager seeds + 5 replays per graph pass; relative L2 ~2.5e-3 (better than the scored path).

Speed (paired cold-L2 medians, event-timed):

| version | 2 experts | 3 experts |
|---|---:|---:|
| scored TINY path | 31.2 us | 39.5 us |
| sk1 v1 (smem RMW down accumulation) | 39.1 | 50.2 |
| sk1 v3 (register down accumulation, x in smem, 8 warps, NS 4) | 37.2 | 47.7 |
| sk1 v3, NS 5 | 41.0 | 50.7 |

Cycle profile (v3, 2 experts, per CTA): prologue 2.8 us (producer waits behind setup), main loop 20.7 us
(gate/up compute 9.0 + wait 4.6, down compute 4.3), flush + barrier ~4.9 us, final norm pass ~4.9 us.
The loop is gate/up-compute-limited (~1.1 us per 36 KB stage at 8 warps; 16 warps hit the 544-thread register
cap and spill). Fixing prologue/final would only reach ~parity with the scored kernel; not pursued further.
Deeper rings were consistently slower here and in the L2-prefetch screen.
