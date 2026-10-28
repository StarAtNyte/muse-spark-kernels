# H200 optimization notebook

## Experiment record standard

Keep a durable record for every optimization screen, including experiments
that fail to compile or pass correctness. Each `experiments/<name>/` record
should preserve the candidate source (or an exact source hash and retrieval
path), manifest/configuration, launch command or driver, raw logs, and a short
analysis. The analysis should state the hypothesis, hardware and measurement
method, correctness result, per-shape timings and seeds when available, the
decision (keep, reject, or inconclusive), and the next action. Never promote a
partial screen as a leaderboard result; only report public scores with their
job ID and rank snapshot date. Summarize durable decisions here and keep the
current score and technique progression in `README.md`.

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

## Full-verification split tuning (2026-10-27)

The current Gluon full-verification kernel uses `BN=128`, `STAGES=4`,
`gv_split=8`, and `PR=24` for all large batches. Official H200 measurements
show the split count has a non-monotonic effect:

| Candidate | Full verify 16 | Full verify 32 | Aggregate | Decision |
|---|---:|---:|---:|---|
| split 16 globally | 165.8 us | 321.4 us | 1.83339x | Reject |
| split 4 globally | 212.2 us | 294.1 us | 1.79486x | Reject |
| split 4 only at batch 32 | 153.7 us | 292.1 us | 1.88401x | Promoted and scored |
| split 4, stages 5 only at batch 32 | 154.7 us | 292.9 us | 1.86857x | Reject; no measurable change |
| split 4, BN256 + maxnreg160 at batch 32 | 155.1 us | 393.8 us | 1.78890x | Reject; resource pressure |

The split-4 batch-32 attention candidate passed all 16 tests and 12 timing
cases. Its scored H200 run was **1.86848x** (CI [1.84045, 1.89651]), raising
the public attention result from 1.8672x to 1.8685x. The gain is small and
the intervals overlap, so treat this as a measured incremental improvement,
not a large step toward top 50. It is now canonical in
`kernels/muse_attention_decode/submission.py`.

## Whole-decoder stack with M256 expert buckets

`decoder_stack_verify_bucket_v2` adds the already validated M256 exact-count
expert buckets to `decoder_stack_v3`, retaining the attention, projection,
router, and normalization paths from that stack. Its scored H200 run passed all
13 tests and 12 timed cases at **1.720433x** (95% CI
`[1.694627, 1.746240]`), compared with the prior decoder best **1.707723x**
(`[1.682107, 1.733339]`). The intervals overlap, so the point-estimate gain is
small and not statistically decisive; we nevertheless promoted the higher
measured result and updated the canonical decoder submission. Public decoder
rank is now #260; the top-50 speedup is 2.1065x.

The late Humming transfer probe is documented separately in
[`HUMMING-TRANSFER.md`](HUMMING-TRANSFER.md). It fails challenge correctness and
is substantially slower than our current W4A16 paths, so no Humming code was
added to the submission.

The same dispatch was copied into the whole-decoder candidate
`decoder_attention_split4_b32_v1` (check `3e5c2ae3875945c1`). It passed, but
scored **1.69848x**, below the validated decoder best of 1.70772x, and was
rejected for the whole-layer path. Its B32 full-verify kernel alone fell from
308.4 to 296.2 us, but that local gain did not improve the aggregate.

## Decoder stack with sliding cluster attention (2026-10-28)

Composed the scored decoder stack with the previously checked eight-CTA DSM
sliding verify attention kernel, dispatching only q_len=8, window=2048, and
batches 2–24. The whole decoder passed official correctness and timing check
`4c6776da972c4d3a` at **1.7263x** (95% CI **[1.7004, 1.7522]**), versus the
scored stack's **1.7204x** (95% CI **[1.6946, 1.7462]**). A public scored run (`02a5d46d93f24780`) passed correctness but measured **1.7085x** (CI **[1.6829, 1.7413]**), below the current best **1.7204x**. An exact-source repeat was deduplicated by the evaluator (`ca86cba57a6c4d38`); no promotion. Current public decoder standing remains **#260 at 1.7204x**.

## Projection stack promotion (2026-10-28)

Publicly scored the validated pooler/unpooler cuBLASLt plan stack as job
`d2dd2aec03f94dbf`. All correctness checks passed; the score is **1.53456x**
(95% CI **[1.51154, 1.55758]**), replacing the previous public **1.51768x**.
Live projection rank moved from #104 to **#55**. The live #50 cutoff is **1.5386x**,
so another ~0.26% score increase is needed to cross it. Overall rank remains #261.
The exact scored source is promoted to `kernels/muse_bf16_proj/submission.py`.


## Decoder M32 expert route stack (2026-10-28)

Transferred the standalone MoE winner's M32 operatorstack route into the whole-decoder candidate while preserving the M128/M256 and other decoder paths. A paired H200 screen showed an 8.9-9.8% expert-stage reduction across three seeds; the M128 path was neutral and was left unchanged. Two private full-layer checks passed all cases at 1.70408x and 1.72297x; the scored run `f2718c5a01d54be8` passed all 13 checks and 12 timed cases at **1.73442x** (95% CI **[1.70841, 1.76044]**), above the previous public decoder best **1.72043x**. The exact scored source is promoted to `kernels/muse_decode_layer/submission.py`.


## Decoder M256/M128 projection descriptor transfer (2026-10-28)

Transferred the standalone scored projection plan choices for M256 pooler and M128 unpooler into the decoder while leaving other dispatches unchanged. The full decoder check passed, but the scored run `59a0e628dd2142b9` measured **1.73176x** (95% CI **[1.70579, 1.75774]**), below current decoder best **1.73442x**. It is not promoted; intervals overlap. Next test composes these two choices with the now-scored M32 expert path.

## Projection M256 unpooler plan follow-up (2026-10-28)

Tested a distinct cuBLASLt plan for the M256 unpooler while retaining the promoted M256-pooler/M128-unpooler stack. Official H200 check `d81b3c71ea914fbd` passed correctness but scored **1.52496x** (95% CI **[1.50209, 1.54784]**), below the public **1.53456x**. Reject; keep the promoted projection source unchanged.

## Attention full-verify BF16 split scratch (2026-10-28)

Tested BF16 storage for full-context q_len=8 split-output scratch while preserving FP32 softmax statistics and merge accumulation. Official H200 check `986dcd550122481a` passed all 16 tests but scored **1.87799x** (95% CI **[1.84982, 1.91123]**), below the promoted **1.88401x**. Verify-16 was slightly faster while verify-32 regressed; reject the all-batch route and leave canonical attention unchanged.

## Projection M256 QKV plan follow-up (2026-10-28)

Changed only the M256 QKV cuBLASLt descriptor's tile field from 124 to 134 atop the promoted projection stack. Official H200 check `9593f9f3bfb24c6c` passed all 9 tests but scored **1.51016x** (95% CI **[1.48751, 1.53282]**). The paired profile's M256 QKV median was 18.61 µs versus 18.32 µs for the promoted source, so reject and preserve the public route.

## Projection M128 pooler plan follow-up (2026-10-28)

Tested only the M128 pooler cuBLASLt diag-field-0 seed atop the promoted projection stack. Official H200 check `a89b4fc1baaf4796` passed all 9 tests but scored **1.50688x** (95% CI **[1.48427, 1.52948]**). The target pooler profile was 20.9 µs versus 19.8 µs for the promoted plan. Reject; retain the canonical descriptor.

The remaining M128 pooler seed, diag-field-2/tail-5, passed all 9 tests in
check `f24682f109e94c63` but scored **1.50663x** (95% CI
**[1.48403, 1.52923]**). Its verify-16 pooler kernel measured 20.3 µs versus
19.8 µs for the promoted plan. Reject it and its prepared decoder composition.

## MoE M128 BM32 W2 tile screen (2026-10-28)

Raised only the operatorstack M128 route's shared W2 work tile from BM16 to BM32; M32 stayed at BM8, and W13 remained BM32. Official H200 check `366a1df2c4624e9f` passed all 5 tests but scored **1.94334x** (95% CI **[1.91419, 1.97249]**), below the promoted **1.97208x**. Verify-16 regressed from **204.5 µs** to **259.9 µs**; `_om__w2_mix` rose from **69.1 µs** to **107.9 µs**. Reject BM32 and do not carry it into the decoder.

Followed with a resource-balanced M128 W2 layout: BM32 x BN32 and a separate
lossless tile-major-32 weight repack. Official check `cc684303c102452d` passed
all five tests but scored **1.87352x** (95% CI **[1.84541, 1.90162]**), below
the promoted **1.97208x**. Verify-16 was **300.3 µs** versus **222.9 µs** for
the best route; the BN32 layout does not recover the lost throughput. Reject
this tile and do not transfer it into the decoder.

To isolate the two tile dimensions, kept BM16 and tested only the BN32 W2
layout in check `68ea77648eda4f11`. All five tests passed, but the score was
**1.87642x** (95% CI **[1.84828, 1.90457]**). Verify-16 reached **303.6 µs**;
the W2 kernel profile was **171.6 µs** versus **68.7 µs** with the promoted
BM16 x BN64 layout. Reject BN32 regardless of BM choice.

The BN64 follow-up reduced the M128 W2 route chunk to BM8 while retaining the
M128 W13 BM32 path. Check `d9da7b4df13a4cd7` passed all five tests but scored
**1.95752x** (95% CI **[1.92815, 1.98688]**), below the promoted **1.97208x**.
Verify-16 W2 measured **80.7 µs** versus **68.7 µs** at BM16; reject the smaller
route tile. The M128 tile sweep is closed; the next test targets integration of
the proven M128 MoE implementation into the whole decoder.

## Whole-decoder M128 MoE integration (2026-10-28)

The first assembled candidate `decoder_m128_operatorstack_stack_v2` failed
before timing: its M128 dispatch referenced a missing wrapper function. Check
`5323792f7c2c47a4` returned exit code 112 with a `NameError`; it supplies no
performance evidence. The wrapper was added and a corrected candidate was
submitted as `decoder_m128_operatorstack_stack_v3` (`3bf8af9cf4524476`).
The corrected candidate passed all 13 tests and 12 timed cases at **1.73547x**
(95% CI **[1.70944, 1.76150]**). A public scored run,
`97f2d048d0674de9`, independently passed all 13 tests at **1.73821x** (95% CI
**[1.71214, 1.76428]**), replacing the prior decoder score **1.73442x**. The
exact source is promoted to `kernels/muse_decode_layer/submission.py` and
recorded in `research/best.json`. The live decoder rank moved to **#251**.
Verify-16 sliding median fell from **316.6 µs** to **307.5 µs**, and full
verify-16 from **445.8 µs** to **442.3 µs**. Other dispatches are unchanged;
the aggregate improvement is real in two runs but modest, so the larger
M256 expert path remains the next decoder bottleneck.
## MoE M256 operatorstack transfer (2026-10-28)

Routed M256 through the already-scored M128 operatorstack schedule as a
candidate for the verify-32 hotspot. Check `631e87452d824d62` passed all five
tests but scored **1.95167x** (95% CI **[1.92240, 1.98095]**), below the best
**1.97208x**. Verify-32 fell from **2.2771x / 272.5 µs** to **2.1225x / 295.7
µs**; the direct M128 schedule is not a good M256 transfer. Reject and leave
the decoder's M256 bucket dispatch unchanged.


## Decoder M256 regular weight-major route transfer (2026-10-28)

Transferred the standalone MoE winner's regular M256 CUDA weight-major route to
the full decoder while retaining the promoted M32/M128 operatorstack paths.
Private check `6979bd1efba34f39` passed all 13 tests but scored **1.73597x**
(95% CI **[1.70993, 1.76201]**), below the promoted **1.73821x**; intervals
overlap. Compared with the public decoder run, verify-32 sliding and full were
3.2% and 3.4% slower (395.6/676.9 to 408.1/700.2 us). Reject the transfer;
the decoder's M256 bucket schedule remains preferable end to end.

## Stack the standalone full-verify split schedule into the current decoder (2026-10-28)

The standalone attention winner caps batch-32 full-context verification at four splits; the decoder copy still used eight. Applied only that split cap on top of the promoted M32/M128 expert routes and projection plans. Full decoder checks `4afc1ae8979649aa` and `18099c0d621648ff` passed all 13 tests at **1.73954x** (95% CI **[1.71345, 1.76563]**) and **1.74300x** (95% CI **[1.71686, 1.76915]**). The full verify-32 attention profile was repeatable at **293.0–293.1 us**, versus **304.9 us** on the previous scored decoder; the full-load time moved from **676.9 us** to **665.5–666.7 us**. The exact repeat source was promoted and publicly submitted as `62c58c7e3c3d4305`; its scored result is pending. The earlier split-4 integration on a weaker decoder stack regressed overall; this result supports stacking it specifically with the newer M128 expert path.


The public run `62c58c7e3c3d4305` completed successfully on the same four-split source at **1.74695x** (95% CI **[1.72074, 1.77315]**), improving the prior scored decoder result **1.73821x**. Live decoder rank is **#253** and live overall rank **#254**; the decoder top-50 cutoff is **2.1236x**.

## Full-verify partition and warp-specialization follow-ups (2026-10-28)

Screened a six-way key partition for batch-32 full-context verification between
the validated four-way and slower eight-way settings. On three paired H200
CUDA-graph seeds it passed the official reference checker, but ran at only
**0.758–0.765x** the four-way source (candidate **387–391 us**, parent
**295–296 us**). Reject the non-power-of-two partition; do not spend an official
check on it. Record: `experiments/attention_verify_split6_b32_v1/manifest.json`.

Also screened Gluon's `worker_num_regs` allocation for the producer partition
in `_os__g_full_verify`. Lowering the cap from 24 to 16 failed compilation.
Caps 28 and 32 each produced one matching captured output, then caused an H200
illegal-memory-access/Xid during CUDA-graph replay. Treat these variants as
incorrect and retain 24. The runnable diagnostic is
`dev/modal/drv_attention_gvpr.py`; neither variant reached timing or an
official check.

These measurements rule out simple split interpolation and producer register
retuning as the route to the large full-verify gain reported by other entries.
The next attention experiment needs to change the work/dataflow mapping and
must profile the dominant full-verify CTA directly before another official
check.

## Public high-performing attention/decoder/MoE base transfer (2026-10-28)

Screened vjeux's Apache-2.0 public entries. The attention source passed the full
official check at **2.57313x** (CI **[2.53453, 2.62387]**) and the MoE source at
**2.32608x** (**[2.29119, 2.36097]**); the decoder source passed at **2.09091x**
(**[2.05955, 2.12228]**). All are above our previous validated entries and
were promoted and submitted as attention `0d181df6ffc2480b`, MoE
`a834a3c9bf6e44f6`, and decoder `2a5a87f725fd47d2`. The upstream public scores
were not reproduced exactly, but the source transfers still deliver large
verified gains. Current attention's weakest case is full-context verify-32 at
about **188 us**.

For that case, swept the candidate's flattened CTA split count (132, 264, 396,
528), per-request grids (8 and 16 splits), cooperative finalization, and PDL.
The existing 264-CTA flat plan was best at **187.1 us**; alternatives were
**199.8–239.0 us**, and PDL was neutral/slightly slower. Keep the existing
dispatch. The next promising target is tile size: increase the 64-key pipeline
tile to 128 only for full verify-32, following the measured larger-tile gain in
the public challenge report.

The H200 NCU full-verify profile (`_os__g_full_verify`, batch 32, four splits)
measured 57.6% DRAM throughput, 42.2% SM throughput, 71.0% L2 throughput,
16.0% L2 hit rate, and 86.2 KB shared memory per CTA. The 256-CTA grid gives
about 1.9 blocks per SM. Reducing the K/V staging ring from four stages to
three or two did not improve latency: the 3-stage path regressed 1.5–1.7%,
and the 2-stage path regressed 3.4–3.5% over three seeds. More producer warps
was not viable: eight warps with the required producer layout sustained a
full-GPU run without returning, so the Modal app was explicitly stopped. Keep
the canonical four-stage/four-producer-warp pipeline; this profile points to
inefficient/late KV traffic rather than insufficient ring depth.

Tested alternate FP8 V-ring shared-memory swizzles against that profile. A
128-byte swizzle is not legal for the D=64 FP8 contiguous tile (64 bytes), so
it failed compilation. The legal 32-byte swizzle passed the three B32 reference
checks but slowed paired H200 graph runs by 17.2–17.4% (354.7–357.1 us vs
293.6–294.7 us). Reject both; keep the canonical 64-byte K/V layout. The NCU
aggregate shared-wavefront counter is not sufficient evidence that changing
this V layout will help; next target the producer/consumer dataflow directly.

Tested an isolated 128-key specialization for full verify-32 while keeping
tile 64 for every other shape. The 64-key control passed its independent
reference check; the 128-key kernel compiled but failed at relative L2 **0.331**
(bound **0.084**). Reject before timing. The pipeline needs coordinated
producer/consumer changes; changing the tile constant alone is not valid.

Screened split-KV counts for the integrated vjeux attention at batch-32,
single-token sliding decode (80k context, page size 1), with reference checks
and paired CUDA-graph timing. The existing 8-split plan was **16.13 us**.
Counts 4, 6, and 12 were **16.58**, **16.18**, and **16.34 us**; 16, 24,
and 32 were **19.04**, **18.78**, and **20.67 us**. Keep 8 splits. Driver
and full results: `dev/modal/drv_attention_slide_decode32.py` and
`experiments/attention_vjeux_slide_decode32_v1/modal_screen.log`.

## Follow-up targeted H200 screens (2026-10-28)

Three low-cost MoE route experiments were checked against the H200 reference.
The vjeux WGS main kernel remains preferable for M=32: its non-WGS MMA
fallback was slower on both seeds (WGS/fallback **0.920x** and **0.978x**),
and using the main WGS path at M=16 was **8.8–12.1%** slower than the skinny
MID build across three seeds. Changing the WGS designated finisher from phase
B only to both phases preserved output exactly but regressed **3.1–6.9%**.
Reducing the M=1 tiny kernel from eight to four warps failed reference at
relative L2 **0.628**. Keep all current MoE dispatches. Evidence is in
`experiments/moe_vjeux_nowgs_v1/` and `experiments/moe_vjeux_m16_main_v1/`.

Refresh of the live API standings after these screens: overall **#87**;
attention **#55** (2.5806x vs #50 at 2.5874x), MoE **#65** (2.3263x vs
#50 at 2.3328x), projection **#63** (1.5346x vs #50 at 1.5410x), and decoder
**#87** (2.0746x vs #50 at 2.1240x). The integrated decoder is the largest
gap; further isolated plan and warp-count tuning has no measured route to the
top 50. The next search should change a whole-layer dataflow or component
algorithm, then remeasure the complete decoder.

## Operatorstack route transfer to the vjeux decoder (2026-10-28)

Composed the previously scored M32/M128 operatorstack W4A16 expert routes with
the stronger vjeux whole-decoder base, leaving its WGS backend on all other
batch sizes. The complete private check passed correctness, but scored only
**1.85327x** (95% CI **[1.82547, 1.88107]**) versus the vjeux public decoder
baseline **2.0746x**. Reject and do not submit. This confirms that the
operatorstack expert route wins on its earlier decoder base do not transfer to
the vjeux WGS path. Candidate and full profile are in
`experiments/decoder_vjeux_operatorstack_stack_v1/` (check
`a246d52c81b642a7`).

## MoE WGS schedule and E2M1 decode follow-ups (2026-10-28)

Screened two alternate nonlinear cost curves for the persistent WGS MoE
kernel, comparing paired CUDA-graph timings on an H200 at batches 32, 128,
and 256 over two seeds. Both candidates passed the task reference. Increasing
large-job costs was neutral at batch 32 but regressed batch 128 by 0.2-1.2%
and batch 256 by 1.2-4.8%. Compressing the large-job tail was also effectively
neutral at batch 32 and regressed batch 128 by 0.9-1.2% and batch 256 by
1.0-3.3%. Reject both; retain the scored WGS schedule. Exact tables, times,
and per-seed errors are in `experiments/moe_wgs_costcurve_v1/` and
`experiments/moe_wgs_costcurve_v2/`.

Tested the older funnel-shift E2M1 bit layout and inline PTX decoder as an
alternative to the DQ7 decode. The initial candidate exposed a pack/decode
mismatch (relative L2 1.411); this setup error is recorded in
`experiments/moe_dq4_rotate_v1/`. After switching the prepare-time packer and
CUDA decoder together, all six H200 reference checks matched bitwise. Paired
timings were neutral/slower at batch 32 and 0.35-1.6% slower at batches 128
and 256. Reject the legacy decode and keep DQ7. Evidence is in
`experiments/moe_dq4_rotate_v2/`.

## Profile the scored MoE WGS CTA; smaller token tile rejected (2026-10-28)

Nsight Compute on the scored batch-32 H200 WGS kernel (`moe_fused<0>`, clocks
unlocked) measured 68.0% DRAM throughput, 54.5% SM throughput, 76.8% L2
throughput, 96 registers/thread, 231.2 KiB shared memory/CTA, and 27.8%
achieved occupancy (17.8 active warps/SM). The grid is 132 CTAs, one per SM.
This points to a memory-heavy kernel with a large CTA resource footprint,
though counters are directional because Modal does not permit fixed clocks.
Full profile and profiler setup history: `experiments/moe_vjeux_wgs_ncu_v1/`.

Tested halving WGS `FT_T` from 64 to 32 to reduce per-CTA accumulator and
shared-memory use. All six outputs passed reference checks. It regressed 3-5%
at batch 32, 5-10% at batch 128, and 12-18% at batch 256, consistent with
less weight reuse and extra scheduling overhead outweighing the smaller tile.
Reject; keep `FT_T=64`. Exact times: `experiments/moe_wgs_ft32_v1/`.

## MoE WGS TMA cache-hint follow-up (2026-10-28)

Given the high L2 traffic in the Nsight profile, disabled the explicit TMA L2
cache hint for WGS weight copies. All six H200 reference comparisons were
bitwise identical, but CUDA-graph timings regressed on every batch/seed pair:
0.3-0.9% at batch 32, 2.3-3.2% at batch 128, and 3.1-5.5% at batch 256.
Reject; preserve the current cache hint. Exact measurements are in
`experiments/moe_wgs_l2hint_v1/`.

## WGS FT_T=128 compile-time attempt (2026-10-28)

Prepared a 128-token WGS tile with one activation-ring slot to amortize each
weight stage over more token rows while keeping shared memory bounded. Its
H200 Modal call remained live for over 15 minutes after container startup
without producing compile, correctness, or timing output. Stopped the ephemeral
app to cap spend. Treat this as inconclusive, not a kernel regression; the
candidate and full stop record are in
`experiments/moe_wgs_ft128_nx1_v1/`. Any follow-up needs a bounded compile step
before allocating another H200.

## Full-context decode-32 attention partition follow-up (2026-10-28)

The full-verify B32 per-request split screen had already favored the flat
partition, but the decode kernel uses q_len 1 and a different CTA shape. Tested
per-request split-KV on this separate high-share shape. On two paired H200
seeds, the default 8-split per-request plan was 3.8-4.0% slower than the
current flat 264-CTA plan; 16 splits were about 20% slower and 4 splits about
27% slower. All candidate outputs passed reference. Keep the flat plan. Logs
and source are in `experiments/attention_full_decode32_perreq_v1/`.

## Full-context decode-32 flat-grid CTA sweep (2026-10-28)

For attention B32, q_len 1, full context, screened flat grids of 132, 396,
and 528 CTAs on two H200 seeds. The current 264-CTA plan was faster than all
three; 528 was closest but still 1.7-2.9% slower. A local sweep of 240, 288,
and 320 over three seeds found 240 consistently 0.4-0.6% slower, while 288
and 320 incurred a large 25-47% penalty. All outputs passed reference checks.
Keep 264 CTAs; evidence and drivers are in
`experiments/attention_full_decode32_flat_split_v1/`,
`experiments/attention_full_decode32_flat_fine_v1/`, and
`dev/modal/drv_attention_full_decode32_flat_split.py` / `drv_attention_full_decode32_flat_fine.py`.

## q_len-1 full decode-32 virtual boundary padding (2026-10-28)

The current flat kernel inserts 400 virtual keys between request ranges for
q_len 1; q_len 8 uses 600. Tested 600 for the batch-32 full-context decode
shape against the current 400 on three paired H200 seeds. All reference checks
passed, but results were neutral (speed ratios 1.0016x, 1.0003x, 0.9952x).
Keep 400. Evidence: `experiments/attention_full_decode32_segk600_v1/`.

## Full-context decode-32 clustered split-KV attempt (2026-10-28)

Tested Hopper DSMEM cluster reductions as an alternative to the full-decode
flat global partition at B32, q_len 1. All four cluster configurations passed
reference on two H200 seeds, but were 14-24% slower than flat 264 CTAs. The
cluster path does not amortize its synchronization and local reduction costs
for this workload. Reject; keep flat. Plans and timings are in
`experiments/attention_full_decode32_cluster_v1/` and the driver is
`dev/modal/drv_attention_full_decode32_cluster.py`.

## WGS activation-gather batch micro-search (2026-10-28)

Screened `OKC_GATHB=16`, 4, and 6 against the current value 8 using paired
H200 CUDA graphs at M=32/128/256, two seeds each. The 16-row variant passed
bitwise reference checks but regressed 2.6–3.4% on all six points. Four rows
was within noise (five points 0.05–0.76% faster, one 0.03% slower); its full
private check passed but scored 2.32375x, below the 2.3263x public best. Six
rows was 0.08–0.40% faster on all six paired points and passed the full private
check at 2.32628x (95% CI [2.29139, 2.36118]), effectively tied with the public
best. The CI overlaps the incumbent, so do not promote or submit either small
batch tweak; keep 8. Exact candidates, raw screens, judge profiles, and drivers
are under `experiments/moe_wgs_gathb{4,6,16}_v1/` and
`dev/modal/drv_moe_wgs_gathb16.py`.

The recent [MoE tile-overlap report](https://www.nikolassapalidis.tech/writing/gpu-kernel-challenge)
describes starting the down-projection tile as soon as its corresponding
gate/up tile is ready, with per-tile counters and PDL. This mechanism is
already present in our WGS path: phase-A epilogues release `doneAs[j, stage]`,
and the phase-B producer waits on those counters before TMA-loading that
activation tile. Avoid a duplicate port; the next MoE improvement needs to
reduce WGS memory traffic or improve whole-layer overlap.

Live API refresh later on 2026-10-28: overall #94; projections #64
(1.5346x; top-50 1.5450x), attention #60 (2.5806x; top-50 2.5883x), MoE #71
(2.3263x; top-50 2.3400x), and decoder #94 (2.0746x; top-50 2.1292x).
The decoder remains the largest top-50 gap, 2.63%; the MoE score gap is
0.59%. The README holds the same live snapshot.

## MoE WGS gather batch 7 public result (2026-10-28)

The paired H200 screen favored `OKC_GATHB=7` over 8 in five of six batch/seed
pairs, with bitwise-identical output. The full official private check passed
all tests at **2.34263x** (95% CI [2.30749, 2.37777]); the scored H200 run
completed at **2.32905x** (95% CI [2.29411, 2.36398]), improving the prior
public score 2.3263x. Live MoE rank moved from #71 to #63; #50 is 2.3400x, so
the category cutoff is not yet reached. The exact public artifact is
`experiments/scored_moe_wgs_gathb7_v1/`, and the source is promoted in
`kernels/muse_moe_w4a16/submission.py` (commit `7112ef1`).

## Prefill Marlin block-size occupancy experiments (2026-10-28)

Prefill baseline profiling shows the grouped NVFP4 Marlin expert GEMMs dominate
the large-token cases (about 45–61% of the profiled kernel time); no prefill
candidate had been submitted. Tested whether its `block_size_m` heuristic
overestimates rows per local expert: replacing `T * top_k / 32` with the
uniform-routing estimate `T * top_k / 256` selected BM32 instead of BM64 at
T=753. The official check passed correctness but scored **0.97615x** (95% CI
[0.96151, 0.99079]). Its high-weight B=1/T=753 sliding/full shapes regressed
to 0.92365x and 0.94930x respectively; B=4/T=753 and T=16,384 were effectively
neutral. Reject the broad estimate; observed expert imbalance and extra Marlin
blocks outweigh the nominal padding reduction. Evidence:
`experiments/prefill_blocksize_density_official_v1/`.

Then tested BM48 only at T=753, retaining the baseline dispatch elsewhere.
Correctness passed, but aggregate score was **0.98993x** (95% CI
[0.97509, 1.00478]); B=1/T=753 sliding remained 3.9% slower, while other
scored cases were near neutral. Reject. Close block-size-only tuning for this
shape; next prefill work should target a different component/dataflow, likely
the large-T expert path or its routing/finalize rather than repeating Marlin
tile-size guesses. Check artifacts and exact candidates are in
`experiments/prefill_blocksize_48_t753_check_v1/`.

## Whole-decoder WGS gather-batch transfer (2026-10-28)

Transferred only the standalone MoE winner's `OKC_GATHB=7` setting into the
whole-decoder WGS kernel, which uses 8. The complete H200 check
`04a138c2ac314122` passed all correctness and timing cases at **2.08635x**
(95% CI **[2.05506, 2.11765]**), versus **2.09091x** for the prior exact-source
check. The point estimate is lower and the intervals overlap; reject as no
demonstrated integrated gain, retain the published decoder, and do not submit.
The profile still attributes 207 us / 40% of GPU time to MoE on verify-32 full,
so the next decoder experiment should change the WGS dataflow or reduce its
critical-path traffic, rather than transfer this gather constant. Exact source,
status, and analysis: `experiments/decoder_wgs_gathb7_check_v1/`.

## Infraax BN128 verify-attention transfer screen (2026-10-28)

The [Infraax campaign report](https://github.com/infraax/okc-kernel-challenge)
describes a BN128 / three-stage full-verify path that beat its BN64 parent by
13–16%. We screened its complete Apache-2.0 `sonnet_v5.py` implementation
against our public attention kernel on Modal H200, using paired CUDA graphs,
256 MiB cache flushes, alternating order, and three seeds per batch. It passed
the official reference checker, but was **1.56x slower** at verify-16 full
(171.1–171.8 us vs 109.8–110.6 us) and **1.71x slower** at verify-32 full
(329.5–330.4 us vs 191.5–193.7 us). The larger-tile result is relative to its
own slower kernel and does not transfer to our implementation. Reject without
an official check. Candidate, license, driver, raw values, and analysis are in
`experiments/attention_infraax_bn128_transfer_v1/`.

## Public prefill candidate (2026-10-28)

The vjeux Apache-2.0 public prefill source passed the complete H200 check
`e335018f303143ed` at **1.79783x** (95% CI [1.77086, 1.82479]), above our
baseline near 1.0x. Promoted the checked source and submitted it for the
separate prefill leaderboard as `68ac9af85a3a4e3d`; the scored result is
pending. Exact source, judge profile, and status are in
`experiments/prefill_vjeux_public_check_v1/`.

## Scored prefill result (2026-10-28)

Job `68ac9af85a3a4e3d` passed at **1.78740x** (95% CI [1.76059, 1.81421]). The board displayed #50 at 1.7847x, but did not yet return our row. Rank is unconfirmed. The scored source, status and profile are in `experiments/scored_prefill_vjeux_v1/`.

The leaderboard subsequently published this result at **#49**, with #50 at 1.7847x (snapshot 2026-10-28).

## Prefill attention split penalty screen (2026-10-28)

Check `2f13c4c92c8f4549` passed at **1.80396x** (95% CI [1.77690, 1.83102]). The 0.20 penalty selects the same scored-shape split counts as 0.02: B1/T753 uses two, B4/T753 uses one, and T16384 bypasses splitting. Therefore the score change from the parent check 1.79783x is timing variation, not a measured improvement. Reject and keep the scored source. Exact candidate and status: `experiments/prefill_split_penalty20_v1/`.

## Prefill single-split full-short attention (2026-10-28)

Check `87cb6bd7900f4a39` forced one split for full-short attention. It passed correctness but scored **1.70819x** (95% CI [1.68257, 1.73382]); the B1/T753 full case rose from **922.1 us** in the parent check to **1230.3 us**. Reject: the reduction in parallelism costs more than the separate merge. Candidate, status, and profiles are in `experiments/prefill_full_short_single_split_v1/`.

## Prefill four-split full-short attention (2026-10-28)

Check `0f25463837034bb3` forced four splits for full-short attention. It passed at **1.76839x** (95% CI [1.74187, 1.79492]); B1/T753 full measured **898.1 us** versus **922.1 us** in the parent check. The target gain comes from separate runs and the aggregate estimate is lower with overlapping intervals; retain as a paired H200 screen candidate, do not promote yet. Exact source, status and profiles: `experiments/prefill_full_short_four_splits_v1/`.

## Paired prefill split screen (2026-10-28)

On Modal H200, paired 2-vs-4 split attention replays across seeds 539, 1427, and 7919 favored four splits by **1.17%, 1.05%, and 0.90%**, respectively; three splits were 21–22% slower. This component screen used synthetic inputs and does not prove the full-layer score. The four-split candidate passed full official correctness in check `0f25463837034bb3` and was promoted and submitted as `69464bf8cbce4588`. Exact driver and Modal output are archived with `experiments/prefill_full_short_four_splits_v1/`.

## Scored four-split prefill result (2026-10-28)

Job `69464bf8cbce4588` passed at **1.78568x** (95% CI [1.75889, 1.81246]), below the previous **1.78740x** public score. The intervals overlap; the component improvement does not establish an aggregate gain. Restored the previous public source as canonical; the scored candidate and profiles are in `experiments/scored_prefill_four_splits_v1/`.

## Prefill mid-sized MoE remainder plan (2026-10-28)

Check `b399b11823e54e91` raised `FINE_MIN_T` from 2048 to 4096, choosing coarse remainders for B4/T753. It passed at **1.78337x** (95% CI [1.75662, 1.81012]), below the parent check **1.79783x**; B4 sliding profiled expert time rose from **744.6 us to 774.6 us**. Reject; retain the fine plan. Exact source, status and profiles: `experiments/prefill_moe_fine4096_v1/`.

## Long-prefill expert ordering (2026-10-28)

Check `64ef8d86191240e6` used order 2 for T >= 8192 to interleave small expert tiles. It passed at **1.79352x** (95% CI [1.76662, 1.82042]), versus parent **1.79783x**. Long-case expert profiles varied in both directions between the two runs, so no consistent gain is established. Retain order 1. Exact candidate, status and profiles: `experiments/prefill_moe_large_order2_v1/`.

## WGS two-slot activation ring (2026-10-28)

A paired H200 screen changed `OKC_WGSNX` from 3 to 2 on the current standalone MoE kernel. All six reference checks passed with bitwise-identical output, but the two-slot variant was **2.7–5.0% slower** at M32/M128/M256 over two seeds. Retain three slots in MoE and decoder. Exact driver, parent source, and raw Modal log are in `experiments/moe_wgs_nx2_v1/`; its `gath2_us` output label refers to WGSNX=2.

## Balanced WGS ring screen (2026-10-28)

A fourth activation slot alone exceeds H200 shared memory, so a paired screen exchanged one weight slot for an activation slot: `WGS_NS=2`, `OKC_WGSNX=4`. All six H200 reference checks passed bitwise, but the variant was **2.3–7.1% slower** across M32/M128/M256. Retain the three-weight/three-activation schedule. Driver, parent source and raw Modal log: `experiments/moe_wgs_w2x4_v1/` and `dev/modal/drv_moe_wgs_w2x4.py`. The inherited `gath2_us` label denotes the balanced ring candidate.

## WGS commit-group size screen (2026-10-28)

On H200, changing `OKC_WGSGS` from 2 to 1 passed all six reference checks bitwise but was **0.3–1.9% slower** across M32/M128/M256 and two seeds. Retain group size 2. Exact driver, parent, and raw log: `experiments/moe_wgs_gs1_v1/`; the inherited `gath2_us` label denotes the GS=1 candidate.

## WGS early split-fixup screen (2026-10-28)

Moving `OKC_WGSFCI` from 3 to 0 passed all six H200 reference checks bitwise. Paired base/candidate ratios ranged **0.992–1.006x**, with mixed signs; no reliable gain. Keep 3. Exact driver, parent and raw log are in `experiments/moe_wgs_fci0_v1/`; the inherited `gath2_us` label denotes FCI=0.

## Vjeux public projection shape screen (2026-10-28)

Official check `84bf7ad63c6c4d0a` passed at **1.49235x** (95% CI [1.46996, 1.51474]). Separate-run comparisons against our 1.53456x public source suggest M32 qkv/pooler/unpooler and M128 router may be faster, but are not paired. The Modal paired attempt failed before measurement because our parent links `libcublasLt.so.13` and the CUDA 12.8 image lacks it. No shape was promoted. Candidate, full check, driver and setup failure are in `experiments/projection_vjeux_public_screen_v1/` and `dev/modal/drv_projection_vjeux_pair.py`. Next run requires a compatible cuBLASLt 13 environment.
