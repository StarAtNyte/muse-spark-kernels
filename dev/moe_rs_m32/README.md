# Row-split pipeline for M32

Frozen baseline experiments/moe_wm_v22/submission.py. Official current-best M32 candidate150.11us versus floor70.76us and baseline202.45us: weakest per-load speedup1.349x.

Compare original M32 nt4/nmax4/nw4/ring1 to existing large-batch row-split nt4/nmax4/nw4/ring4, without changes to weights, activations, arithmetic or routing. CPU ptxas identifies original ring1 gate/up 120-byte stores/132-byte loads spilled, down120/128,128registers; row-split gate157/down160 registers, no spills. More shared-memory staging and fewer resident CTAs may still lose; only paired cold-L2 timing decides. Six seeds, independent reference checks and bit equality.

Bundle probe.py is diagnostic-only and supports M32 configs. Driver ../modal/drv_moe_rs_m32.py. GPU test waits until the active stall-profile job terminates, maintaining one GPU job at a time. CPU compile succeeded: compile.log.

## Outcome

First run passed independent correctness but stopped on an unnecessarily strict bit-equality diagnostic. Original ring1 distributes the reduction differently, so rounding can differ. Corrected diagnostic retains the independent reference check and requires relative difference from baseline<0.001. Completed retry screen_v2.log: 6/6 reference checks passed, baseline-relative errors0.00049–0.00073. Speedup ratios0.8943/1.0081/0.9771/0.9851/1.0102/0.8273. Reject due to inconsistent performance and severe regressions on two routings; no official quota spent. Spill removal alone does not overcome this path's staging/scheduling costs at M32. All jobs terminal.
