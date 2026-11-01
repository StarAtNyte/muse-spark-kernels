# SC6 + rs128 merge into our decoder: FAILS correctness, do not submit

v1 (job 7b94d6fb754a4eed) failed with NameError `_qkv_compact_route`: the 2,046-line merge hunk replaced our tail
(compact QKV T=128 route and compact O-layer). v2 restores that tail (984,342 bytes) and runs, but job
e417dee2a9874dc1 passes only 8/13 tests: `batch=3 ctx=6000 dist=outliers layer=full q_len=8` residual update relative L2
1.047 > 0.15. T = 24 is in the 17-32 token WGS range, so the SC6 scale packing or the stage layout does not match the
KPERM3 expert layout our decoder already carries, i.e. the SC6 hunks and the weight re-layout disagree somewhere in the
host-side packing (`_sc6_wgs` vs our `_kperm_*` / `_wgs_copy` order) or in the consumer decode.
The standalone MoE merge (experiments/moe_stack_sc6_v1) was never checked either. Safe alternatives measured on the judge:
keell0renz decoder as-is 2.264x check / 2.243x scored; our decoder 2.226-2.260x. rs128 alone on our decoder: verify-16
+1-2%, whole file within noise (experiments/decoder_ours_rs128_v1).

## Root cause (static diff of merged file vs keell0renz decoder)
The SC6 hunks I selected by keyword (`SC6|sc6`) leave the WGS weight-stage producer on `UNIT_BYTES` (36,864 B) for the
per-stage global stride and the TMA `expect_tx` size, while the SC6 build uses `WGS_STG` (35,840 B). That hunk and ~60
others across the WGS producer/consumer region (old lines ~5300-6800: st1 wp2s gating, stage addressing, host packing)
do not mention SC6 in their text, so the keyword filter skipped them. With SC6 packed weights, every stage after the first
is read from the wrong offset -> garbage experts at 17-32 tokens (residual error 1.05).
Completing the merge means taking essentially all of their WGS region, i.e. their decoder plus our compact-QKV tail;
their decoder as-is already exists (check 2.2636x, scored 2.243x). Conclusion: no separate "ours + SC6" decoder to build.
