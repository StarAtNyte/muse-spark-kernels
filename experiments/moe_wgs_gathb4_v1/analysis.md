# WGS activation-gather batch 4

Hypothesis: the current 8-row gather batch might hold more temporary state
than needed. This candidate changes only `OKC_GATHB` to 4; exact source and
paired H200 screen are archived here.

All six H200 reference comparisons were bitwise exact. Relative timing gains
(base / candidate) were +0.758%, -0.031%, +0.344%, +0.117%, +0.250%, and
+0.045% for M=32, 128, and 256 over two seeds. This is a small mixed effect,
not enough to promote from a microbenchmark. A single full private check will
determine whether the change improves the scored aggregate. Driver:
`dev/modal/drv_moe_wgs_gathb16.py` with `GATHB=4`.

The full private check passed (job `80016f149b434e86`, 5/5 tests, 2.32375x,
95% CI [2.28889, 2.35861], 5.787 ms). It does not clear the current public
2.3263x result, and its confidence interval overlaps. Reject; do not submit.
The judge's exact timing/profile artifacts are in `profiles/`.
