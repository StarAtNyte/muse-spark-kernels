# MoE W4A16 — worker w1 notes (2026-10-30)

Parent: published `scored_moe_ours_kperm3_v1` (job `190d8914bacb4aa2`, 2.3820x; check `2ecdb5bf891d40e4` 2.3911x),
source SHA-256 `ac008155…792f` (= `kernels/muse_moe_w4a16/submission.py`). Leader dolanzhao 2.7578x (needs +15.8%).

## Per-case gap (check 2ecdb5bf891d40e4, judge µs after its ~4 µs graph subtraction)

| load | build | ours | judge floor | ours/floor | practical floor* | ours/practical |
|---|---|---:|---:|---:|---:|---:|
| decode 1 (bs1) | TINY | 24.47 | 11.42 | 2.14 | ~15.3 | 1.60 |
| decode 16 | MID | 91.00 | 52.52 | 1.73 | ~63.5 | 1.43 |
| decode 32 | WGS | 121.40 | 70.76 | 1.72 | ~85 | 1.43 |
| verify 1 (bs8) | TINY2 | 67.23 | 37.32 | 1.80 | ~45.7 | 1.47 |
| verify 16 (bs128) | WGS | 176.14 | 101.34 | 1.74 | ~120.6 | 1.46 |
| verify 32 (bs256) | WGS | 208.23 | 108.94 | 1.91 | ~129.5 | 1.61 |

*practical = floor bytes / 4.1 TB/s (best measured H200 TMA stream, `experiments/h200_read_methods_v1`) + ~2 µs ramp.
Every case is 1.43–1.61x off a pure TMA stream; the leader's +15.8% would put them at ~1.25–1.4x. The gap is
spread evenly across loads, i.e. it is the streaming skeleton, not a single shape.

## Tried this session (Modal H200, paired cold-L2 CUDA-graph medians, 30 alternating replays, full reference check)

### 1. L2 lookahead prefetch of the producer's own weight stages — REJECTED (large regressions)
`cp.async.bulk.prefetch.L2` of the stage D ahead of the TMA ring (exactly the bytes the CTA will read next, no
extra traffic). Source `findings/w1-work/cand_wpf.py` (`OKC_WPF`, `OKC_WPFH` evict_last hint, `OKC_UPF` tiny/WS
builds), driver `findings/w1-work/drv_variants.py`, logs `wpf_screen.log`, `upf_screen.log`. All outputs passed.

| bs | D=3 | D=6 | D=6 evict_last |
|---:|---:|---:|---:|
| 32 (WGS) | 0.950 / 0.951 | 0.910 / 0.888 | 0.892 / 0.869 |
| 128 (WGS) | 0.925 / 0.932 | 0.863 / 0.860 | 0.866 / 0.870 |
| 256 (WGS) | 0.921 / 0.918 | 0.821 / 0.830 | 0.858 / 0.866 |
| 1 (TINY) | 0.880 / 0.885 / 0.892 | 0.813 / 0.798 / 0.804 | |
| 8 (TINY2) | 0.925 / 0.922 / 0.915 | 0.851 / 0.855 / 0.841 | |
| 16 (MID) | 0.948 / 0.940 / 0.917 | 0.877 / 0.843 / 0.838 | |

(gain = base/candidate; seeds 403,419 for WGS; 401,404,402 for tiny.) Monotone: more lookahead = slower, at every
size. Together with the earlier PFB (whole-expert prefetch), ring-depth and NS4/NX2 results this rules out
"too few bytes in flight" as the limiter: adding outstanding requests beyond the 3-slot ring makes HBM slower.
Next-hypothesis implication: the loss is in DRAM access efficiency/ordering or in the per-stage handshake, not depth.

### 2. Final-pass ablation (upper bound for early/overlapped finalize)
See below (pending at time of writing).
