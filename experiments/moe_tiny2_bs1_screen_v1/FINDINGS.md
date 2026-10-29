# MoE: TINY2 build for bs 1 (TINY_MAX_M 4 -> 0)

Build sweep `experiments/moe_build_select_v1` (bs1: TINY2 1.086x, MID 1.077x vs TINY; bs8 TINY2 stays best). Paired screen here (`dev/modal/drv_moe_pair.py`, 6 seeds, all outputs reference-checked incl. bs 2-5 normal/skewed): bs1 36.16 -> 33.53 us (1.078x), bs 8-256 0.999-1.002x.

## Judge result: no gain — lab/judge discrepancy

Check `77b1ba6ab0844072` passed 5/5 at 2.3257x; decode-1 25.5 us vs 24.9 us in the scored parent, and the judge
profile shows the bs1 kernel at 29.8 vs 29.4 us. Re-screened on the judge's exact decode-1 seed (220742048650789979
and +1..5, 2-3 experts; `experiments/moe_build_select_v1/judgeseed*.log`) on both the CUDA 12.8 and CUDA 13 Modal
images: TINY2 still 1.091-1.094x faster in the lab. The judge's H200 runs the TINY build ~2 us faster than Modal's
while TINY2 matches, so this lab win does not transfer. Not submitted. Lesson: confirm small MoE wins on the judge.

The decoder version (`experiments/decoder_tiny2_screen_v1`) showed an artifact: when base and candidate share
`load_inline` module names but run different builds, the base slows down (A/A control `aa.log`: 1.000-1.012x).
Screen decoder variants that change builds in separate processes.
