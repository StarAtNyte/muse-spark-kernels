# H200 optimization notebook

## Objective and evidence

Optimize the geometric mean of six load speedups, where each load is the
weighted sum of its layer times. Do not optimize the mean of shape speedups.
For each candidate retain the exact source, SHA256, hypothesis, official job,
JSON results, profiler reports, and a decision. Only a full passing run of the
integrated candidate establishes its performance; selecting the best measured
shape from several runs is an experiment proposal, not a verified score.

Method reference: [Shui et al., 2026](https://arxiv.org/html/2607.17979v1).
The useful transfers are workload-based routing, bounded hypotheses,
profiler-guided iteration, archival, and correctness-gated promotion.
Their B200 kernel techniques and measured gains are not H200 predictions.

## Constraints that determine the design

- H200 SXM, cold L2 before every timed graph replay. Repacking is allowed in
  untimed prepare; activation work stays inside kernel_fn.
- BF16 dense weights cannot become FP8 or FP4. Router FP32 weights may use
  BF16 high/low splitting. Preserve BF16 activations at required boundaries.
- KV is FP8; attention queries may be FP8. Attend to every visible key.
- Public kernel rankings establish serving priority; cash prizes are on the
  later serving track. Favor optimizations that compose into SGLang.
- Official checks have quotas. Reserve five projection checks for integration
  and correctness fixes rather than spending all twenty on tile sweeps.

## Projection round 1: measured

`bf16_splitk_v1`, job `3f1954adb1044e31`: all 9 tests and 30 timed cases
passed. Speedup **1.035754x**, interval **[1.020217, 1.051290]**.

The load score recomputed from official per-shape records matches the server.
See `experiments/bf16_splitk_v1/analysis.md` for timings and profiler reports.

Diagnosis:

- Router fusion helps most: 1.432x at decode 16, 1.454x at decode 32,
  2.576x at verify 1, and 1.379x at verify 16. Verify 32 is only 1.037x.
- `o` at decode 32 is 0.590x and unpooler is 0.729x: the universal split-K
  route is inappropriate there. Restoring GEMM is a justified candidate.
- Pooler at small batches gains about 0–9%; qkv is near parity or worse.
- Profiling reports a separate reduction kernel on every custom route.
  Removing this launch is plausible where the main launch already has enough
  independent output tiles. Underfilled launches still require splitting.
- The ideal load-level roofline ceiling relative to this run's baseline is
  approximately 1.83–2.61x. This omits fixed overhead and cannot be treated as
  achievable performance. Dense projection tuning alone has limited upside.

Decision: **archive only**. Correct, modest aggregate gain, material shape
regressions. No public submission or claim of being competitive yet.

## Projection rounds 2 and 3: measured

`bf16_gemv_v2` passed all 9 tests and 30 timed cases; aggregate **1.031684x**,
interval **[1.016208, 1.047159]**. Dense one-token speedups are 1.127x qkv,
1.110x o, 1.201x pooler, and 1.192x unpooler. The direct router GEMV is
**0.576x**, negating those wins. Decision: **archive only**, retain dense
singleton route as an integration candidate, exclude its router route.

`bf16_transposed_v3` passed all 9 tests and 30 timed cases; aggregate
**0.845123x**, interval **[0.832446, 0.857800]**. Router speedups by token
count are 1.531x (1), 3.614x (8), 2.163x (16), 2.088x (32), 1.676x (128),
and 1.360x (256). Small pooler batches improve about 9–11%. Larger dense
GEMMs regress severely: verify-32 qkv 0.300x, pooler 0.345x, unpooler 0.293x.
Decision: **reject globally**, retain router and small pooler routes as
integration candidates. Contiguous transposition and a CTA-count heuristic
alone are insufficient for dense GEMM; reusing too little weight data across
small M tiles is a plausible limiter, requiring hardware profiling to confirm.

## Active bounded experiments

| Candidate | Hypothesis | Official job |
|---|---|---|
| bf16_gemv_v2 | One-token vector FMA reduction avoids tensor-core padding, scratch writes, and merge launch | 22bdb0294f8c4816 |
| bf16_transposed_v3 | Contiguous K,N weights, larger K tile, occupancy-based split count, direct output when split=1 | 6128e66277164905 |
| attention_fused_io_v1 | Fuse QK norm, interleaved RoPE, FP8 Q conversion and scaled KV writes before FA3 | 9ba7233d35414c89 |
| bf16_hybrid_v4 | Integrate v1 router, v2 singleton dense, restore cuBLAS on regressing dense shapes | ab09c58d874a41e0 |
| bf16_hybrid_v5 | Integrate v3 router and small pooler with v2 singleton dense; cuBLAS elsewhere | 8c1cc35a00be4e05 |
| attention_splitkv_v2 | Exact sliding-window split-KV with GQA reuse and fused merge, BF16 rounding, norm and gate; FA3 for full attention | c29a12701caf42ee |
| moe_baseline_profile | Profile the unchanged pinned Marlin baseline before selecting MoE changes | c761669d797940e1 |
| attention_all_splitkv_v3 | All-key paged split-KV for full contexts with dynamic GPU loop bounds; retains GQA reuse and fused merge/norm/gate | 314d01393cc24fa4 |
| decode_baseline_profile | Whole-layer baseline profile to identify critical path and overlaps | f7759c9480d447f2 |
| decode_fused_router_v1 | Test fused router inside the complete layer, preserving FP32 small-batch routing and stream overlap | 2674829a37f040b2 |
| attention_all_splitkv_v4 | Fix FP8 mask-value typing; same all-key split-KV algorithm as v3 | eb6de29d0a314bef |
| moe_fused_small_reduce_v1 | Bypass small-batch PyTorch normalization/reduction fallback with the fused Triton kernel | 7c48aa5ad79c4601 |
| moe_marlin_m16_v2 | Use M=16 expert-alignment tiles at larger batches to reduce skewed-route padding | 8bf6b00f456e4952 |
| moe_local_align_v3 | Align mapped local IDs using 32 buckets and omit invalid-route blocks | 2382392969184ca3 |

These are hypotheses until full official results arrive. Keep the first
candidate unchanged while its alternatives are evaluated.

## Validated integrations and scored submissions

`bf16_hybrid_v4` passed all 9 tests and 30 timed workloads, **1.103890x**,
95% interval **[1.087331, 1.120448]**. Promoted as the current best projection
implementation, replacing v1 in `kernels/muse_bf16_proj/submission.py`.

`attention_fused_io_v1` passed all 16 tests and 12 timed workloads,
**1.133564x**, 95% interval **[1.116561, 1.150568]**. Promoted as the current
best attention implementation. Full-context FA3 dominates larger batches:
approximately 187 us at 16 requests and 374–497 us at 32 in the profiler.
Preprocessing fusion alone cannot remove this dominant cost.

At the user's instruction to keep submitting, scored runs were submitted:

- Projection hybrid v4: `b42a0772e39546b1`.
- Attention fused I/O v1: `c701dd346ce94bdc`.

Those are new scored jobs, not published results yet. Their full records are
tracked under `experiments/scored_*`; `research/submissions.json` records
submission provenance. Continue submitting verified improvements within caps.

`bf16_hybrid_v5` subsequently passed all 9 tests and 30 timed workloads at
**1.168898x**, interval **[1.151364, 1.186431]**. It improves the validated
integration from v4; no dense shape has a material regression against SGLang.
Router speedups are 1.544x (1 token), 3.670x (8), 2.076x (16), 2.031x (32),
1.733x (128), and 1.373x (256). The full six-load aggregate, not those router
ratios, determines its 1.168898x score. V5 is promoted to the projection
submission and submitted for scoring as **`3ee947021eae43b9`**. The scored
job remains pending; the private score is not yet a public ranking result.

## Serving preparation

`okc serve init serving` fetched the pinned SGLang checkout at `a55ab7d5`.
The 3,828-byte patch adds `fused_split_router.py` and routes the existing
Hopper split-BF16 router through it for 9–256 tokens and 256x8192 weights.
It uses the existing hi/lo tensors; the FP32 small-batch router and larger
prefill paths stay intact. Python syntax, git whitespace, allowed file paths,
size, and line-length checks pass. This is not an 8-GPU validated serving win.

The existing serving attention code already has optional fused QK norm and KV
stores, unlike the standalone challenge baseline's unfused invocation. Do not
assume the standalone preprocessing score will transfer unchanged to serving.

## Attention compilation failure and local compiler preflight

`attention_splitkv_v2` finished with `passed: false`, return code 112.
The outer result says `status: ok` because the evaluator ran successfully;
this must never be interpreted as kernel correctness. Triton cannot cast an
integer `other=0` directly to `fp8e4nv` in masked loads. The full-context v3
shares that source issue and remains unpromoted. Candidate v4 uses `0.0` at
the three FP8 masked loads, preserving the intended zero semantics.

A Python 3.12 virtualenv now contains Triton 3.6.0 (no torch installation).
`tools/compile_attention.py` extracts the JIT functions and compiles them
directly for SM90 without needing an H200. It reproduces the original v2
failure as a negative control; v4 passes ten preprocessing, attention, and
merge compilations spanning decode/verify, full/sliding, scaled/unscaled and
page-size paths. Metadata is in its `compile-sm90.json`. This proves code
generation only; remote correctness and performance are still required.

```bash
.venv/bin/python tools/compile_attention.py experiments/attention_all_splitkv_v4/submission.py
```

## MoE baseline profile: measured, next three isolated probes

`moe_baseline_profile` passed 5 tests and 6 timed workloads. Its **1.001456x**
is parity within the interval, not an optimization. The useful artifact is
`experiments/moe_baseline_profile/analysis.md`.

- End-to-end baseline latency by token count: 77 us (1), 150 us (8),
  188 us (16), 205 us (32), 462 us (128), 617 us (256).
- Below 32 tokens the supplied post-expert wrapper deliberately executes a
  PyTorch fallback. Its many elementwise and reduction launches account for
  roughly 30–40 us. The two Marlin calls use about 41 us at one token. Probe
  v1 executes the same fused Triton norm/reduction used for larger batches.
  Its SM90 code generation passed locally; runtime correctness is pending.
- At 128/256 tokens Marlin accounts for 95–96% of profiled GPU time. The
  stock tile heuristic chooses M=48/64, based on a global route-count estimate.
  Probe v2 tests M=16 without changing routing or the reduction path. Less
  padding under skewed local routing is a hypothesis, not a measured win yet.
- Local expert IDs are already mapped to [-1, 31], but the baseline asks the
  alignment helper to build 256 expert buckets. The pinned helper supports
  32 buckets and `ignore_invalid_expert=True`. Probe v3 isolates this change:
  invalid routes are not local computation, and downstream activation/norm
  masks them exactly as before.

All three probes keep the model's selected routes, weight precision, and
normalization. Do not combine them until their individual full checks pass.

## Completed results and next integration round

Scored results now confirm projections v5 at **1.174299x** and fused
attention v1 at **1.130524x**, both with all correctness and timed cases
passing. The observed leaderboard positions were 184/359 and 232/258
respectively; this is progress, not a winning result.

New validated private results:

- MoE fused small reduction: **1.146258x**, all 5 tests and 6 timing cases
  passed. The one-token route improves **1.477x**, 8 tokens **1.262x**, and
  16 tokens **1.214x**. Promoted and scored submission **15fdc2d1955549b9**
  is queued.
- MoE M16 tiles: **1.109286x**. Verify 16 improves **1.382x** and verify 32
  **1.339x**, with smaller batches at parity. These disjoint regime wins are
  integrated with fused small reduction in **moe_hybrid_v5**, check
  **c0b80be362af4fcf**. Local alignment v3 is at parity and is not integrated.
- Whole-layer fused pipeline: **1.097519x**, all 13 tests and 12 timing cases
  passed. Promoted and scored submission **4026b7baf4e04652** is queued.
  Router-only whole-layer improvement was just 1.013x, confirming that
  router/pooler overlap hides much of the standalone router speedup.
- Projection WGMMA v6: correct but **1.068458x**, worse than v5. Hardware
  WGMMA selection is necessary for some routes but insufficient for a win;
  leave the validated cuBLAS routes in place. Compiler and cubin reports are
  in `research/projection-mma-layouts.json` and `projection-registers.json`.
- Attention all-split-KV v4: correct but **1.019025x** aggregate. Decode-1
  sliding improves **1.940x**, full **1.398x**; large verify full-attention
  falls to 0.54–0.56x. Hybrid **attention_hybrid_v5**, check
  **bbd722c637134459**, retains only promising small-sliding and decode-full
  routes, using FA3 elsewhere.

The next architecture probes are native MoE **769e726bc944456a** and native
FP8 QK attention **ca9e327f05c04b3d**. Native MoE decodes all 16 E2M1 codes
exactly into BF16 bit patterns, multiplies the original FP8 block scales,
applies the FP32 global scale after accumulation, and preserves BF16
activation boundaries. Gate/up and SwiGLU are fused. Local SM90 compilation
confirms WGMMA and zero stack allocation with BN32; the earlier BN64 probe
had a 232-byte stack and was revised before official submission. Its high
register count remains a performance concern to measure.

FP8 QK uses Q and K already stored in FP8 directly in their tensor-core dot
product, avoiding the prior lossless widening to BF16. Softmax probabilities
and the value product retain the existing BF16 path. Both probes are pending
remote correctness/performance and have not been promoted.

`tools/promote.py` now enforces full expected case counts, passing flags,
source hashes, and aggregate improvement before copying a candidate to the
canonical kernel. `--submit` records a scored source snapshot and job ID.
File locking serializes best/history updates, and existing scored manifests
prevent duplicate launches. Negative controls reject the failed compilation
candidate and the slower WGMMA candidate before making any network mutation.

## One-token MoE and small-batch dense probes

The scored MoE v1 result confirms **1.146519x**, with all coverage passing.
`moe_singleton_v7`, job **52f4f669d02647e1**, adds packed-FP4 vector GEMVs
only for the one-token regime. The first stage fuses gate/up, BF16 SwiGLU,
and local-ID mapping. It keeps FP32 global scales and the same expert norm.
The fallback combines the independently validated small-reduction and M16
routes, whose integrated check is still pending. SM90 compilation reports
128 registers for gate/up, 95 for down, and zero stack allocation for both.
This is compilation evidence only; correctness and timing remain pending.

`bf16_skinny_v7`, job **6dc8a03e41354b9b**, tests row-major weight loads with
one BM32 tile over 2–32 tokens. Its split count is lower for wide output
matrices than the original uniform split-8 probe. Singleton GEMV, router,
and larger-batch cuBLAS paths retain the current best implementation.
Five representative and correctness shapes compile for SM90.

The current **5,021-byte** serving draft additionally enables the existing
fused expert norm/reduction for the validated small-batch BF16 Hopper shape
(8 routes, width 4096, weighted pre-norm, explicit slot IDs). Other shapes
retain their original dispatch. Syntax, whitespace, allowed paths, patch
size, and added-line limits pass. The combined serving patch still needs
8-GPU runtime validation; do not report its component gains as a serving win.

Use `python tools/experiment.py collect --pending` to poll live jobs without
reprinting terminal results. Terminal checks can still be explicitly
re-fetched with `collect NAME` when needed.

The scored whole-layer result has now passed at **1.096193x**, interval
**[1.079750, 1.112636]**. Together with projections, attention and MoE, this
establishes scored entries on all four decode-related boards. It does not
establish serving priority or a winning position; those require the actual
overall ranking and ultimately the serving-track results.

## Native MoE rejection and TMA projection branch

Native MoE v4 completed full correctness but scored **0.580715x**. Verify-16
is 2,554 us versus 456 us for SGLang; verify-32 is 2,944 us versus 624 us.
The profiler attributes roughly 1,570 us to gate/up and 930 us to down.
Dispatch and output reduction are negligible in this candidate. Decision:
**reject globally**. Hardware WGMMA selection alone did not make this a
competitive quantized GEMM. Its 234–254 registers and per-element decoding
remain plausible limiters, but a profile without hardware counters does not
prove the relative importance of occupancy, decoding, padding, or layout.

`bf16_tma_v8`, job **b20bdf3f300043ef**, replaces register-staged loads in
larger projection batches with host TensorDescriptors and TMA loads. It uses
row-major weights, 64x128 output tiles, and the existing FP32 split reduction.
The host descriptor path needs no custom runtime allocator. SM90 compilation
confirms TMA and WGMMA instructions, 90 registers, and zero stack allocation
across four representative/odd-row shapes. Small projections and the router
keep the validated v5 paths. Actual graph compatibility, correctness, and
timing are pending the official check.

The prefill baseline is now queued to establish its true critical path before
changing prefill GEMMs. Prefill does not count toward overall kernel rank,
but its latency matters for the serving track's time-to-first-token bound.

## MoE integration improvement

`moe_hybrid_v5` passed all 5 tests and 6 timed workloads at **1.265151x**,
interval **[1.246174, 1.284128]**. It combines the independently validated
fused small reduction and M16 large-batch tiles. Per-load speedups are
1.468x, 1.217x, 0.999x, 1.263x, 1.365x, and 1.332x in the official order.
It is promoted to the canonical MoE submission and scored job
**33168954e1684bad** is queued. Its private score is not yet a scored result.

`decode_hybrid_v3`, check **717244d81a0744f5**, applies the validated M16
choice inside the fused whole-layer pipeline for 32–512 tokens. The serving
draft now includes a matching Hopper/BF16/32-local-expert guard for this tile
choice; the draft is **6,011 bytes**, passes syntax and packaging rules, and
still needs full serving validation. Prefill above 512 tokens keeps its
original Marlin tile selection.

`moe_m8_midbatch_v8`, check **9d1bc7f8348d46d3**, probes M8 tiles only in the
32–64 token band. This addresses the one remaining scored MoE load at parity
in the current hybrid. The lower and larger batch routes remain those of v5.
This is a performance hypothesis until its official result arrives.

## Attention hybrid promotion and packed MoE revision

Attention hybrid v5 completed all 16 tests and 12 timed cases at
**1.238320x**, interval **[1.219746, 1.256895]**. It is promoted and submitted
for scoring as **2c0559f533d145dc**. The custom sliding decode-1 route is
1.973x over SGLang, while larger verify loads retain the faster FA3 route.

Native FP8 QK v6 passed all coverage at **1.175689x** overall. It is not a
global improvement over v5, but its full decode times improve from about
186/357 us to 157/301 us at 16/32 requests. Large verify remains slower than
FA3. **attention_fp8_hybrid_v7**, check **6e7717bfd5e04527**, integrates the
native FP8 QK wins, uses custom single-token decode when useful, and sends
large verify or long-context verify blocks to FA3. SM90 preflight passed ten
kernel variants; official correctness and timing are pending.

**decode_attention_hybrid_v4**, check **0dd0d5d2f7aa4430**, composes the
validated v5 attention dispatcher with the fused layer and M16 Marlin route.
It preserves the whole-layer candidate's BF16 QK rounding and leaves FA3 on
the same losing regimes. Its integrated layer result is not yet established.

The native MoE revision **moe_packed_v9**, check **a6e5098b3c3445b0**, follows
the packed decoding approach seen in the pinned Marlin dequantization source.
It loads eight FP4 codes in one uint32, expands them with inline PTX to BF16,
and applies the original block scales. Eight output values are ordered by a
join tree whose lane permutation was checked explicitly. Weights remain
packed FP4 in memory; the int32 view is only a word-sized view of the same
bytes. FP32 global scaling and BF16 activation boundaries are unchanged.

This revision also uses BM16 to reduce expert padding. SM90 cubins show
136 registers for gate/up and 105 for down, both with zero stack, versus
254/234 in the rejected original. This does not prove a performance win.
Official validation and timing must still pass before any promotion.

Singleton MoE v7 also finished: correctness passed, but the one-token route
fell to **0.551x** (145 us versus 80 us for stock), making its aggregate
**1.079605x**, below hybrid v5. Its profile shows 62 us in gate/up and 48 us
in down; normalization is only 2.9 us. Decision: **reject globally** and retain
Marlin plus fused reduction for this regime. The scalar decoding GEMV did not
achieve its expected bandwidth advantage. Its failure reinforces the need to
measure packed decoding rather than assume fewer matrix-padding operations
will translate into lower latency.

## Next decisions after results

1. Check correctness before interpreting any timing. Scaled FP8 caches,
   page-size differences, outliers, and odd batch sizes are important cases.
2. Compare each shape against the paired SGLang result in its own run.
   Hardware variation makes cross-run absolute latency selection unreliable.
3. Integrate only clear shape-regime wins; test the entire merged dispatcher.
4. For remaining dense bottlenecks, inspect tile waves, repeated weight reads,
   FP32 reduction traffic, shared-memory use, and registers before tile sweeps.
5. For attention, inspect how much FA3 versus preprocessing costs. A custom
   sliding-window path could fuse output normalization into the final merge;
   long contexts need efficient paged gathers and KV reuse across query heads.
6. For MoE, the pinned baseline uses Marlin W4A16, routing alignment, a separate
   SiLU multiply, and normalized weighted reduction. Investigate routing and
   expert-major packing before attempting a replacement quantized GEMM.

## Hopper-specific reference boundaries

[NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
documents TMA and the occupancy tradeoffs that should guide further experiments.
[Triton persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
is a reference for persistent/TMA routes. Its CLC route requires compute
capability 10+, so it is not an H200 route. Do not copy Blackwell TMEM or
native FP4 assumptions into Hopper kernels.

## Development commands

```bash
python tools/experiment.py collect
python tools/analyze.py experiments/bf16_splitk_v1
python tools/experiment.py check NAME path/to/submission.py --hypothesis 'one bounded change'
```

The launch helper parses Python without executing GPU imports, snapshots code,
checks remaining quota, submits privately, and immediately records the job ID.
It does not promote or make public submissions. After an ambiguous network
failure, inspect `okc jobs` before retrying the launch.

For a development H200 with the challenge image, a separate local benchmark
tool validates normal/outlier inputs and times paired graph replays with a
256 MiB flush before each replay:

```bash
python tools/bench_projection.py experiments/bf16_hybrid_v5/submission.py --layer router --m 16
```

This helper has passed syntax and CLI checks but has not been GPU-executed
here. It reports raw graph event timings, without official overhead removal,
and is not an official scorer. Official remote checks remain authoritative.
