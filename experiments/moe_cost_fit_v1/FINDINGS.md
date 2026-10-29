# Fitted WGS stage-cost tables — accurate fit, no speedup (2026-10-29)

Least squares of per-CTA phase time on per-n-tile stage counts (`dev/modal/drv_moe_cost_fit.py`, bs 32/64/128/256,
4 seeds): phase A R^2 0.995, const 13.1 us, 1.29..2.15 us per stage for n-tiles 1..8 -> table
112,126,131,144,155,163,172,187 (scored 112,135,140,158,173,185,200,207); phase B R^2 0.983 -> 100,111,120,129,138,
145,156,171 (scored 100,104,115,123,132,144,156,174).

Paired screen (`dev/modal/drv_moe_cost_fitted.py`, 4 seeds, reference-checked, log `fitted.log`): fitted A only
0.975-0.989x, fitted A+B 0.987-0.994x at bs 32/128/256. Balancing per-CTA phase times does not shorten the kernel:
late phase-A CTAs do not idle the machine (others stream phase B), so the visible tails redistribute work rather than
waste bandwidth. The WGS kernel is throughput-bound at ~3.3 TB/s overall; neither math (compute ablations) nor
scheduling (this) is the lever. Remaining gap to the ~3.9 TB/s TMA probe is in the streaming skeleton itself.
