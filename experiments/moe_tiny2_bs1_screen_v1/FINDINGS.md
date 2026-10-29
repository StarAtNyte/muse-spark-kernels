# MoE: TINY2 build for bs 1 (TINY_MAX_M 4 -> 0)

Build sweep `experiments/moe_build_select_v1` (bs1: TINY2 1.086x, MID 1.077x vs TINY; bs8 TINY2 stays best). Paired screen here (`dev/modal/drv_moe_pair.py`, 6 seeds, all outputs reference-checked incl. bs 2-5 normal/skewed): bs1 36.16 -> 33.53 us (1.078x), bs 8-256 0.999-1.002x.
