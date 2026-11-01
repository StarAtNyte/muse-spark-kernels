# Cold-L2 instruction warming of the cluster epilogue (attention): small positive, not submitted

`cluster_finish` is now one out-of-line function with a `dry` flag (no cluster barriers, no remote loads, no stores);
a dry call runs in `segment()` right after the first cp.async batch is issued, so the epilogue code is fetched while the
first key tiles are in flight (technique from Kushagrabainsla A13: cold-L2 flush makes once-per-CTA code come from DRAM).
Check 91f56b318b0b4904: 16/16 passed, 2.616x vs 2.589x for the unchanged file in the same window.
decode 1 sliding 5.1 -> 5.0 us, decode 16 sliding 8.6 -> 8.0 us, verify 32 sliding 14.7 -> 14.5 us, others unchanged
(+-2-4% is noise). Our epilogue was already kept short ("kept short, runs from a cold cache"), so the saving is ~0.1-0.6 us,
not their ~2 us. attention_warm_plus_route_v1 adds the keell0renz module for sliding verify at batch 1 (3.431 -> 3.752
paired): check 2.617x, 16/16, equal to warm-only. promote.py refused it (not above the best validated 2.6177x).
Next candidates from the same idea: a 5th "warmer" warp for the prologue-adjacent code in attnc-style kernels, and the MoE
(Kushagrabainsla measured warm code saving 0-4% there).
