# WGS MoE (bs 32) instruction and stall profile (2026-10-29)

ncu on the scored kernel (`dev/modal/drv_moe_wgs_ncu.py`, logs here): 105 us, DRAM 66.9% (344 MB read, 3.27 TB/s),
52.4 M warp instructions (ALU 20.8 M, FMA 18.4 M, uniform 4.4 M) -> ~2.4 lane-instructions per FP4 weight, schedulers
~57% issue-active (0.57 issued/cycle, 4.45 active warps, 1.29 eligible). Stalls per issue: long_scoreboard 1.39,
not_selected 1.26, gmma 0.82, math_pipe_throttle 0.79, wait 0.75, barrier 0.65.
Interpretation: the kernel is instruction-heavy (decode + scale multiplies) with modest occupancy, not DRAM-starved;
fewer instructions per weight is the lever, not more bytes in flight.
