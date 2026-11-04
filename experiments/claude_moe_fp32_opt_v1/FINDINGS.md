# claude_moe_fp32_opt_v1: 4-stage weight ring with 16 KB x slots at bs 32 (rejected, 1.6% slower)

## Verdict
Rejected. One paired H200 run on the judge image: at bs 32 the candidate was **0.9838x** the scored FP32 parent (geometric mean of six per-seed paired medians; every seed 0.970-0.995, i.e. slower). The unchanged-path controls (bs 16, 128, 256) came out 1.002, 1.003, 0.998; these are a limited control, not a confidence interval. All outputs passed the reference check. No submission was made.

## Hypothesis and derivation
Weighted target: bs 32 is the worst scored shape against SGLang (1.687x in the official profile, 125 us vs a 70.8 us idealized reference estimate). The bandwidth figure in research/H200-LOWER-BOUNDS.md uses modeled traffic, not a measurement of HBM throughput; weight streaming is a hypothesis to test.

Prior ring experiments did not test this exact point:
- NS=3/NX=2 and NS=2/NX=4 (moe_wgs_nx2, moe_wgs_w2x4) traded one ring for the other: slower.
- NS=4/NX=2 with the routing array overlaid (moe_ring_overlay_v1): 4.3% slower, but it cut the x ring to 2 slots, so it could not separate weight depth from x depth.
- FT_T=32 + NS=4 (moe_ft32ns4_v1) failed ptxas (96-register cap) and had the barrier-index overlap.

Observation: at bs <= 32 no expert has more than 32 tokens (a token routes to an expert at most once), so an x slot needs only 32 rows = 4 n-tiles = 16 KB, not 32 KB. Keeping `FT_T`/`WGS_NT` at 64/8 leaves every register array and the wgmma variants untouched (candidate ptxas: 96 regs, 0 local, same as parent). Rings become 4 x 36 KB + 3 x 16 KB = 192 KB (parent 204 KB), so no overlay trick is needed, and x depth stays 3.

Candidate also relaxes the "no 3-stage burst" throttle `n >= 2 ... pn = n - 2` to `n >= WGS_NS-1 ... pn = n-(WGS_NS-1)`: identical for NS=3, and for NS=4 it keeps one slot for the consumer while allowing 3 weight stages (108 KB/SM) in flight instead of 2 (72 KB). Rough bandwidth-delay estimate: 4.8 TB/s x about 1.5 us / 132 SM is about 55 KB, so the parent is not obviously under-covered; this was the part I was least sure about.

## Change (four text edits inside `CUDA_SRC` plus one table entry)
`make_candidate.py` applies them to the scored source (sha256 `2c6ebcb7...432b96`, verified) and writes `submission.py` (sha256 `ae038dfc...a16501`):
1. `WGS_NS` from `OKC_WGSNS` (default 3).
2. `WGS_XS = (OKC_WGSXR/8)*4096` (default `OKC_FT_T`, i.e. unchanged), with a static_assert.
3. Barrier indices derived from the ring sizes (`BW_E=NS, BX_F=2NS, BX_E=2NS+NX`); for NS=3 they equal the old 3/6/6+NX.
4. Throttle expressed in `WGS_NS`.
5. `PS_SHAPES[32] = ('main', {OKC_WGSNS: 4, OKC_WGSXR: 32}, None)`: only bs 32 gets the new build. The tiny-build copy of the text (`VAR_SRC`) is untouched. FP32 partial sums, stream-K fixups, routing, norm and combine are unchanged; weights stay NVFP4.

Safety contract: the 32-row slot is valid only while M <= 32; it is selected by exact M == 32. The skewed bs 32 tests put all 32 tokens on one expert (slot exactly full) and passed.

## Measurement
One Modal invocation (`run_modal.py`, H200, judge image, cpu=16, timeout 300 s; app ap-o757V4JIpL6hTXZJMCh3Sl); the driver took 130 s of its 255 s budget (parent prepare 128 s with the candidate builds compiled in parallel, then checks and timing). Raw output: `modal_raw.log`. Hashes: `hashes.txt`.

Timing: cold-L2 (512 MB flush) CUDA-graph replays, parent and candidate alternating order each repetition, 40 reps per bs-32 seed and 24 per control seed, empty-graph time (3.14 us) subtracted, per-seed value = median of paired ratios parent/candidate (>1 means candidate faster).

| bs | seeds | per-seed paired median | geomean |
|---|---|---|---|
| 32 | 419, 7919, 15241, 3, 77, 1001 | 0.9814, 0.9950, 0.9848, 0.9697, 0.9909, 0.9811 | **0.9838** |
| 16 (control, same build) | 101, 202 | 1.0040, 0.9997 | 1.0018 |
| 128 (control) | 404 | 1.0029 | 1.0029 |
| 256 (control) | 505, 606 | 0.9961, 0.9998 | 0.9979 |

Absolute bs 32 times (parent -> candidate, us): 117.1->119.0, 133.2->133.8, 126.5->128.5, 78.6->81.4, 129.3->130.4, 111.0->113.7.

Correctness: 13 extra cases (bs 1, 4, 17, 24, 30, 31, 32 x4 incl. two skewed, 33, 128 skewed, 257) plus every timed case, checked before and after the timed replays: all candidate outputs within tolerance (relative L2 about 2.4-2.6e-3 vs the 1.5% bound). Parent-vs-candidate bitwise equality was true in 8 of 13 cases and false in 5, including bs 1 where both use the same TINY build. Run-to-run nondeterminism is plausible, but the cause of each difference was not isolated.

## Interpretation (what is and is not shown)
- Shown: on these six seeds the 4-stage / 3-in-flight ring is consistently a little slower at bs 32 (about 1.6%), larger than the control noise; the 75th-percentile paired ratio is below parity for all six seeds.
- Not separated: the deeper ring and the relaxed throttle were changed together, so I cannot say which one costs the time. Not measured: memory-system occupancy, L2 behaviour, per-CTA timelines. One plausible reading (untested) is that three weight stages in flight add HBM/TMA contention without removing the real stalls, which earlier notes attribute to per-unit tails and instruction issue (57% issue-active), not ring depth.
- This is another data point against ring depth as the bs-32 lever: NS/NX trades and now a pure weight-depth increase with unchanged x depth are all neutral or negative. The remaining gap is more likely instruction count (about 2.4 lane-instr per FP4 weight) and unit-end tails.
- Single Modal GPU, standalone MoE only, not the judge's routing/router; the official judge may differ by a few percent. A 1.6% gap is near the judge's minimum +/-1.5% CI, but the sign is consistent across seeds.

## Files
`make_candidate.py`, `submission.py` (candidate), `driver.py`, `run_modal.py`, `modal_raw.log`, `hashes.txt`. Reproduce: `python3 experiments/claude_moe_fp32_opt_v1/make_candidate.py experiments/scored_moe_fp32_ybuf_v1/submission.py experiments/claude_moe_fp32_opt_v1/submission.py && modal run experiments/claude_moe_fp32_opt_v1/run_modal.py`.
