# MoE cold-vs-warm code cost, repeated (2026-11-01)

moe_icache_share_v1 method, REPS=40, each bs measured twice: bs16 0.6% / 1.2%, bs32 7.0% / 6.9% (100.2 us cold vs 93.2 us
warm), bs8 0.7% / 0.7%. Reproducible, but only bs=32 (WGS build, our weakest MoE load). bs128/256 (same WGS code) show
0% / -2.5%, so "cold instructions" alone cannot explain it.

moe_phase_code_cost_v1: per-CTA stamps at bs16 (MID build): cold 102.1 us vs warm 97.3 us end to end; routing is 2.6 us in
both, the 4.8 us gap accumulates by the end of the first part of phase A (+3.5 us at "gathered") and +1.2 us after: the
first pass through the main compute loop, not once-per-CTA prologue/epilogue code. The attention-style dry pass (A13)
therefore does not apply; the analogue would be executing the decode+MMA loop body on dummy shared memory while the first
weights are in flight. The bs=32 WGS debug build crashed (illegal memory access, cause unknown), so no phase split for it.
Prize if fully recovered: ~7 us of the 32-token MoE case = ~+1.2% MoE board, ~+0.6% decoder (below both top-50 lines).
Not built.
