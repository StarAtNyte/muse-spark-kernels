# Decoder investigation — w4

Baseline: exact scored decoder b72f02c7eb8745f5, 2.2603208825x, 95% CI [2.2264160693, 2.2942256957]. Source in experiments/scored_decoder_compact_qkv_o_v1 matches canonical at investigation start. No public submission unless new official CI lower bound exceeds 2.2603x.

## Evidence reviewed
- TASK.md, AGENTS.md, reference computation/checking and current decoder dispatch; current-decoder.json is an older snapshot (2.0746x), not the current baseline.
- ENGINEERING.md and H200-LOWER-BOUNDS.md critical-path and decoder tables: overlapping profiler durations must not be summed as removable latency.
- decoder_full_stack_m32_lt_v1 is an old duplicate, not a fresh pending candidate (manifest job 71abb5bce1834203).
- decoder_vjeux_operatorstack_stack_v1 profile: 1.85327x; verify16 sliding 408.64us, Marlin expert calls 306.7us. Therefore blanket stack composition can lose by changing expert backend.
- decoder_moe_floor_diag_v1 is useful routing/floor evidence but uses an older parent and is not the current critical path.

## Changes and screens
Candidate w4-work/stack.py changes GATHB 8 to 7 in both embedded MoE sources and installs _TUNED7[(16,1,True,1)] = {n_splits:8,coop:0,nc:1}. Preserves compact T128 QKV/O and decoder-specific B4 finalization. Source-level standalone/decoder MoE diff archived under findings/w4-work/moe_source.diff.

First CUDA-only H200 screen failed before execution: missing tvm_ffi. Retained failure log in experiments/decoder_w4_gath7_plan16_screen_v1/modal.log. Rerunning with public lmsysorg/sglang:dev-muse-spark-open image and a bounded 1100-second process timeout. Screen compares two seeds on all 12 scored shapes, reference checking each candidate and baseline, 24 alternating cold-L2 graph replays.

Boundary screen uses one prepared state to isolate launch/glue changes: preload1 vs2, carry4 vs8, interleave2 vs1, finish4 warps, route2 parts, serialized T16/32 vs PDL. Each variant reference-checked; kernel event profiles collected separately from timings. No arithmetic/weight precision changes introduced.

## Completed initial screens
Stack: all 24 baseline/candidate reference pairs passed, decode32 sliding gain 1.0054–1.0088x, others approximately neutral. This does not establish an aggregate improvement. Official check launched via tools/experiment.py into decoder_w4_gath7_plan16_v1.
Boundary: 64 checks passed, 16 seeded timing rows, 32 kernel profiles archived. Disabling PDL consistently slower at T16/32 (1.5–6.6% latency increase). Preload1 gains 0.3–0.9% at decode32 sliding and 0.06–0.19% full, but loses at smaller T. Other geometry/interleave variants mixed or slower; reject broad changes.

## B4 extension capacity proof
Initial large-M B4 screen failed safely at host TORCH_CHECK: M <= 32. Original B4 uses the final 64 rows of ybuf for partial norms, enough for 1024 routes x256 floats. For T256 worst-case 2048 routes, reserve 128 rows instead: 128*4096 = 2048*256 floats. EMAX=384*8=3072; live ybuf rows end at 2048, partials start at 3072-128=2944. Thus the regions are disjoint, and fixup storage begins after row3072 unchanged. Candidate stack_b4_capacity.py changes guard to M<=256, guard route count <=EMAX-128, and scratch offset to EMAX-128. Screen v2 tests T128/256; production routing remains unchanged until evidence supports enabling the variant.

## Final results and decisions
Official check 56bffe6cfc6a4eae passed 13/13, speedup **2.1978738907x**, 95% CI **[2.1649057823, 2.2308419991]**, score_us 10291.9008. Submission gate failed. **No public submission or promotion.** The current incumbent remains unchanged. Note the incumbent CI overlaps this check slightly; this is not a clean causal proof that these two code changes regress, but there is no demonstrated win.

Weighted paired local estimate was 1.000466x (46:16 within each load, then geometric mean over six loads and two seeds), using rounded printed latencies. This is effectively neutral and is not a statistical confidence interval.

B4 capacity extension: all 16 baseline/variant checks passed, eight timing pairs; variant was slower in every pair, by 1.04–3.60us. Reject extending B4 beyond T32. Eight kernel event profiles archived. The initial host-guard failure is retained separately.

All boundary-sweep checks passed (64), and all initial stack checks passed (48 total including baseline). Boundary sweep includes 32 kernel event profiles; B4 includes eight. Official baseline and candidate per-case profiles and latencies are in findings/w4-work/{baseline_profiles.json,candidate_profiles.json,official_comparison.md}. Event rows can overlap and include PDL waits; conclusions use whole graph timings.

No new unverified RMSNorm/projection algebra or router/top-k cooperative fusion was submitted. Existing decoder already has the T1 projection/residual normalization decomposition, fused pooler/router, top-k plus expert-input normalization, and PDL chaining. Tested safe geometry/launch variants did not establish an improvement. The modest T32-only preload1 screen signal (0.3–0.9% sliding) remains unpromoted; it needs independent confirmation before another check.

No commits, no canonical project edits, no public submission. All candidate changes are confined to new experiment directories and requested run notes.

### Official before/after case times

| Case | Incumbent us | Candidate us |
|---|---:|---:|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1 | 68.11 | 69.22 |
| batch=1 ctx=80000 dist=soft layer=full load=decode 1 page=1 | 74.58 | 75.54 |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic | 127.79 | 130.40 |
| batch=16 ctx=80000 dist=soft layer=full load=decode 16 page=1 spread=agentic | 206.25 | 207.80 |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode 32 page=1 spread=agentic | 148.24 | 151.47 |
| batch=32 ctx=80000 dist=soft layer=full load=decode 32 page=1 spread=agentic | 292.06 | 304.00 |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify 1 page=1 q_len=8 | 107.26 | 110.89 |
| batch=1 ctx=80000 dist=soft layer=full load=verify 1 page=1 q_len=8 | 114.39 | 117.81 |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify 16 page=1 q_len=8 spread=agentic | 237.47 | 239.06 |
| batch=16 ctx=80000 dist=soft layer=full load=verify 16 page=1 q_len=8 spread=agentic | 324.60 | 321.55 |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify 32 page=1 q_len=8 spread=agentic | 311.13 | 303.35 |
| batch=32 ctx=80000 dist=soft layer=full load=verify 32 page=1 q_len=8 spread=agentic | 470.73 | 472.63 |
