# w3.2 router rk_kernel M=128/256 — working notes (2026-10-30)

Files: rk2_mod.py (rk2 = generalized rk: NWG x 64 rows, BN tokens, optional pre-packed W via 1-D bulk copy, ablation modes;
rk3 = int8-lo variant), rk_orig.cu (vjeux RK_SRC verbatim, extracted from scored_projection_vjeux_hybrid_v1), sub_bundle.py
(module + ORIG_SRC as uploaded), drv_rk2.py (correctness via reference.check_implementation + paired cold-L2 graph timing),
screen*.log (raw Modal H200 output). All times are event-timed graph replays after a 256 MB flush, median of 41-61 paired
interleaved reps; TINY_GRAPH_US (graph with one trivial kernel) = 5.22-5.38 us, subtract it to get kernel time.

## Facts established
- rk already IS split-K (S=12 at M=128 -> 96 CTAs, S=8 at M=256 -> 128 CTAs) and already runs the hi and lo wgmmas on the same
  staged x tile. The assignment's suggested change is the status quo.
- rk timings in this harness: M=128 10.8-11.1 us (≈5.6-5.8 us kernel), M=256 12.1-12.4 us (≈6.9-7.1 us kernel).
- Ablation (screen2/4/6, rk-identical tile 64x64): load-only (no MMA, no red, no barrier wait) is ≈4.7 us at M=128 (10 MB unique),
  ≈6.2 us at M=256 (12 MB), ≈4.35 us at M=64 (9 MB, W pre-packed contiguous, each byte read once). So a ~10 MB cold read runs at
  ≈2.1 TB/s effective here; rk is only ≈1.0-1.1 us above its own load-only time. The 2.2/2.7 us "floors" (4.8 TB/s) are not
  reachable at this transfer size; marginal cost ≈0.35 us/MB.
- Bigger CTA tiles (2-4 consumer WGs, BN 128/256, more split-K) are uniformly slower (screen1): red.add bytes per tile grow and
  per-SM work rises; e.g. M=128 [2 WG,BN128] S=32 +2.3 us, M=256 [2,128] S=32 +2.2 us. No config beat rk at M=128; best at M=256
  was [1,128] S=16 at -0.13 us (tie).
- Pre-packing W into contiguous per-CTA tiles (1-D cp.async.bulk instead of 2-D TMA boxes) gives at most ~0.2 us, within noise.
- int8-lo (rk3): lo = q*2^(E_hi-142), q int8 → W 8 MB -> 6 MB. Correct on all checks (weight rel rms 6.6e-6 vs 2.4e-6 for BF16
  hi+lo; output rel L2 well inside 1e-4). But the in-smem expansion costs 2-5 us (inline or in a dedicated converter WG; no spills
  per ptxas). Upper bound with expansion skipped (wrong values, timing only, screen8): +0.48 us at M=128, +0.38/+0.61 us at M=256.
  So even a free expansion cannot clear the 1 us bar.
- My rk2 skeleton is consistently 0.1-0.7 us slower than rk at the identical tile config (unexplained; not the x/W TMA order nor the
  extra bar.sync).

## Verdict
No candidate wins by > 1 us (best real gain ≈ 0; best theoretical, with free int8 expansion, ≈0.5 us). No official check run,
nothing submitted, no project files modified. The router M=128/256 lever is worth at most ≈+0.1-0.2% of the score.
