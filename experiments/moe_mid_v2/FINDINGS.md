# moe_mid_v2: MoE bs16/bs32 floor, timeline and ablations (no candidate)

Base: scored FP32 MoE, sha256 2c6ebcb7...432b96 (`base.py`). Two Modal H200 rounds (`r1_abl.log`, `r2_knob.log`).

## Floor
One touched expert = 3*4096^2*(1/2+1/16) B = 28.31 MB. T_floor = route (~2-2.5 us) + D*28.31 MB/4.2 TB/s + tail, where D = distinct local experts (judge generate_input, 200 seeds).
- bs16: D mean 11.6 (5-17): 328 MB -> 78.1 us; judge 90.7 us (86% of floor).
- bs32: D mean 16.2 (8-24): 458 MB -> 109.0 us; judge 125.0 us (87%).
Instruction floor: 7 ALU + 2 IMAD + 4 HMUL2 per 8 weights. ALU alone ~48 us at bs32 = 44% of the BW floor, so decode is not binding. An exhaustive bit-layout search found no decode under 5 ALU ops.

## Timeline (globaltimer per CTA, bs32 with 12 experts: 100.4 us vs 80.9 floor)
Startup ~3.5 us; phases A/B stream at 3.75-4.0 TB/s; phase-B finish spread 4.6 us; final combine 3.3-3.8 us.
bs16 (MID, 13 experts, 101.4 us): startup 2.8-4.6 us (half the CTAs finish routing 1.8 us late), B waits ~2.2 us per CTA for A, phase B at 3.66 TB/s, spread plus barrier ~4 us.

## Ablations (wrong outputs, so upper bounds; >1 = faster)
Remove the A->B waits: bs16 0.973, bs32 1.004. Remove the FP4 decode: 0.989 / 1.037. So FMA-pipe decode is capped at about 3.7% at bs32 and gives nothing at bs16.
Knobs: bs32 cost U 0.996-0.998; finisher 0.994/0.966; bs16 tail and NS4 0.996/0.989/0.977. All at optimum.

## Remaining lever
The phase-B spread plus the final pass (~8 us at bs32) needs a structural change. moe_finpar_screen_v1 (norm sums in phase B) gave +1.2% at bs32 but lost at bs128/256; gating it to bs<=32 is the best known lever, and it is still under 3%.
