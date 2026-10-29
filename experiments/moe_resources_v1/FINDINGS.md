# MoE kernel resource usage (2026-10-29)

Driver `dev/modal/drv_moe_resources.py` (cuobjdump on the scored builds). WGS (NW=16): 96 regs, STACK 0, LOCAL 0. TINY/TINY2: 149 regs; MID (SPLIT): 153 regs; no spills anywhere. The WGS long-scoreboard stalls (`experiments/moe_wgs_stalls_v1`) therefore come from global loads in the dependency waits (ld.acquire polling of xready/doneAs) and stream-K fixups, i.e. cross-CTA dependency waiting rather than spills or decode math.
