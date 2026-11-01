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
