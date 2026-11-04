# Batch-one attention finalization group size

Changed `fin_nc1` from two to four grouped passes when four are available, keeping FP32 QK/PV and workspace calculations intact. The first two-seed paired H200 screen passed four reference checks and suggested 0.3–1.7% verify improvement, but the unaffected decode path varied similarly.

A fresh five-seed confirmation (`../attention_fin_group4_confirm_v1/modal.log`) passed all ten checks. Verify speed ratios were 1.00916, 1.00455, 1.00304, 0.99700, 1.00302 (geometric mean about 1.0034×). Decode ratios ranged 0.9854–1.02416 despite the change not applying to its single pass, showing substantial measurement drift. Event timings include harness overhead. This is insufficient evidence of a gain; retain the parent. The experiment covers one batch and sliding shapes only, not all attention variants.
