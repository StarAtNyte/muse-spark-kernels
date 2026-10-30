# KPERM3 technique stacked on our own GATHB=7 WGS MoE

Port only the KPERM3 hunks (lossless K permutation, one scale pair per four k16 steps) from the Apache-2.0 alygg77 repo onto our public 2.329x source (scored_moe_wgs_gathb7_v1), without their helper-warp MID source or context-bind additions. Tests whether the technique alone transfers to our base. Expect ~+1-2% (their reported +1.25%).
