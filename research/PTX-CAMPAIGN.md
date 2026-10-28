# H200 campaign toward top-50 overall, October 26, 2026

The user authorizes aggressive Modal H200 testing/profiling and exploration of
inline PTX, CUTLASS, and CuTe DSL. Official correctness, precision, and timed-work
rules remain authoritative. Existing uncommitted MoE development files belong
to earlier work; this campaign uses archived validated sources as its parents.

## Current gap

Authenticated live boards at the start of this campaign:

| Track | Our score | 50th score | Our rank |
|---|---:|---:|---:|
| Dense projections | 1.1821x | 1.5166x | 678 |
| Attention | 1.5440x | 2.3696x | 401 |
| MoE | 1.9270x | 2.2015x | 183 |
| Whole decoder | 1.5143x | 1.9613x | 312 |

Overall rank was 316. Track cutoffs are diagnostic targets, not a formula for
overall rank. Whole-decoder parity with today's 50th entry requires about 23%
less time than our scored decoder. Rankings move as competitors submit.
The top 50 overall receive serving scheduling priority, per the official rules;
the rules do not describe this as exclusive qualification for the next round.

## Research examined and applicable hypotheses

1. [Humming, October 19, 2026](https://vllm-project.github.io/2026/10/19/humming.html)
   and its [source](https://github.com/vllm-project/humming): choose pipelines by
   regime; small batches can favor cp.async and warp MMA over larger asynchronous
   matrix pipelines. Transfer: preserve W4A16 and tune activation reuse, weight
   movement, and dequantization. Do not import W4A8/W4A4 precision changes.
2. [Dissecting GPU Utilization for LLM Inference on Nvidia Hopper, September 2026](https://arxiv.org/abs/2609.12923):
   distinguish fragment utilization, occupancy, waves, stalls, and memory traffic.
   Transfer: measure these before assuming WGMMA improves small batches.
3. [KernelPro, revised August 2026](https://arxiv.org/abs/2606.26453): use focused
   Nsight/SASS evidence to select experiments and prune failed hypotheses.
4. [Cloudrift modern matmul series, May 2026](https://www.cloudrift.ai/blog/gpu-matmul-optimization):
   independently measure staging, vectorization, split-K, and asynchronous-copy
   changes. Its principal measurements are large square RTX 5090 GEMMs; numbers
   are not H200 decode predictions.
5. [GPU MODE lectures](https://github.com/gpu-mode/lectures) and
   [resource stream](https://github.com/gpu-mode/resource-stream): primary lecture
   materials for tensor cores, profiling, SASS, and CuTe. Use as engineering
   references rather than evidence of a win on this workload.
6. [FlashAttention-4 author writeup](https://tridao.me/blog/2026/flash4/): SFU and
   shared-memory pressure can matter independently of tensor throughput. Transfer
   the bottleneck analysis; its TMEM/UMMA and two-CTA techniques target Blackwell.
   A linked author tweet could not be fetched directly (X returned 403), so no
   tweet-specific technical claims are used.
7. [FlashMLA Hopper deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md):
   account for conversion costs relative to tensor-core throughput. Sparse-key
   selection is prohibited here; only data movement/conversion insights transfer.

## Evidence and active probes

### Attention Nsight

`dev/ptx_campaign/attention_ncu_unlocked.log`: batch 32 full-context decode,
the validated `fd3_partials<2>` path on Modal H200:

- Duration: 211.81 us under profiling.
- DRAM throughput: 64.52%; L2 throughput: 78.55%; compute throughput: 37.26%.
- 139 registers/thread, 84.48 KB dynamic shared memory/block.
- Shared memory limits residency to two CTAs/SM; theoretical occupancy 12.5%,
  achieved occupancy 11.94%.
- The first NCU attempt could not lock GPU clocks. `--clock-control none`
  resolved it. These are diagnostic counters, not official scored latencies.

### Attention merge

`dev/ptx_campaign/attention_merge_v1.py` starts from validated `attn_wg5_v1`.
Compute the intra-CTA softmax merge factors once per query row rather than
repeating four exponentials for every one of 64 output dimensions. Synchronize
before output threads consume those factors. Preserve output rounding and
every attended key.

Initial paired Modal comparisons across batches 3, 16, 32, and 33, two fresh
seeds each, show 1.061–1.078x over the validated parent, with identical outputs.
The initial reference-check invocation passed a tensor where the local checker
expects `reference.collect(data, output)`; corrected in the second driver run.
Official private integration job: `ac952a38225f454c`, archived under
`experiments/attention_ptx_merge_v1`.

### Attention occupancy

`attention_stage1.py` adds a synchronous single-slot cp.async path for fd3.
It deliberately sacrifices load/compute overlap to reduce shared-memory usage
and potentially improve residency. Sweep split counts against the validated
parent on Modal. No performance claim until the sweep finishes.

### MoE explicit PTX activation cache

`dev/modal/drv_moe_ptx.py` compares `ld.global.ca.v4.u32`, its
`L1::evict_last` variant, and a lower minimum-residency setting with the original
`__ldcg` activation loads. Weight loads, FP4 representation, and BF16 boundaries
stay unchanged. Compare two seeds per scored batch, plus odd/skewed checks.

This diagnostic image has no SGLang, so the driver uses the submission's own
Triton finish kernel for all batches on both sides. A complete official check
must still validate the unmodified SGLang reduction/fallback integration.
The first attempt stopped at batch 32 because the imported SGLang reduction was
stubbed; its partial small-batch results establish no improvement. The corrected
run is `dev/ptx_campaign/moe_cache_v2.log`.

## Completed campaign results

The attention merge passed all 16 official tests and 12 timed cases. Private
speedup was 1.573111717x; public job `3659bd2bb1bb471f` scored 1.584431479x,
up from the previous public 1.5440x (about 2.6%). The validated source was
promoted to `kernels/muse_attention_decode/submission.py`. Last observed overall
rank was 323 and attention rank 374; top 50 has not been achieved.

Other official checks were rejected for promotion despite passing correctness:

- Shared-memory XOR swizzle plus split-policy change: 1.558637x, below the
  simpler merge. Archive: `attention_swizzle_split_v1`.
- Decoder integration of the merge: 1.508117x, below its validated 1.516715x.
  Archive: `decoder_ptx_merge_v1`.
- Router tuning: all shapes selected the existing implementation; its score
  difference is not evidence of an algorithmic improvement.

The completed attention stage sweep found no benefit from single-stage staging.
Use `attention_stages_v2.log`; the earlier run included invalid non-power-of-two
split counts for the Triton reduction.

MoE activation-cache experiments produced aggregate ratios of 0.997685x for
L1 caching, 1.001579x for evict-last, and 0.972730x for reduced minimum residency.
None warrants an official check. Corrected evidence: `moe_cache_v2.log` for
L1 caching and `moe_cache_v3.log` for the remaining variants.

The completed dispatch sweep (`moe_dispatch_v2.log`) rejected ring modes 5 and
6 for incorrect outputs/unsupported shapes. Most correct variants regressed.
Batch 128 configuration `(4,4,4,2)` improved 0.9–1.7% on two seeds; a separate
six-seed confirmation at batches 128 and 256 is recorded in
`moe_dispatch_confirm.log`. Local comparisons use the common native Triton
finish described above, so they cannot establish official integration success.

### CUTLASS CuTe DSL WGMMA

The official rules list both CUTLASS and CuTe DSL as available. We exercised
NVIDIA's Hopper dense GEMM example with swapped operands, split-K, FP32 partials,
and BF16 final outputs. Standard tiles mostly regressed. Thin N=16/32 tiles with
occupancy 2 showed up to approximately 5% on QKV batch 16 and 3% on batch 32,
but did not establish a broad aggregate gain. Forcing three pipeline stages did
not improve this result. No CuTe candidate was promoted.

Valid logs: `cute_projection_v3.log`, `cute_projection_tuned_v2.log`, and
`cute_projection_stage3.log`. The first `cute_projection.log` is INVALID: its
explicit CuTe stream escaped CUDA graph capture. The corrected driver captures
on the same explicit nondefault stream and changes the input after capture,
then replays and checks the new output. The second run failed default-stream
capture. Never use either initial run as speedup evidence.

### Dispatch confirmation outcome

The six-seed confirmation finished successfully: all 12 comparisons produced
identical outputs, but 10 were slower. Median speedup was 0.984323x at batch 128
and 0.974309x at batch 256. Reject `(4,4,4,2)`; keep the validated MoE dispatch.
No official check or public submission was spent on this change. Structured
evidence: `dev/ptx_campaign/moe_dispatch_confirm.json`. The Modal app completed
and stopped normally.

## Verification-attention campaign continuation

Refreshed authenticated board: overall rank 325, attention rank 375, public
attention 1.5844x, leader/ours ratio 1.9134. Snapshots are
`research/current-overall.json` and `research/current-attention.json`.

Two isolated Modal probes start from the current promoted attention source:

- `drv_verify_pipeline.py`: stages 2/3/4 and split counts 8/16/32/64, batches
  8/16/32 with two fresh seeds, paired cold-cache CUDA graph comparisons.
- `drv_verify_exp.py`: replace half or all probability exponentials with
  degree-five FP32 polynomial evaluation after nearest-integer range reduction.
  Extreme values retain the original SFU path, including masked infinities.
  Inspired by FA4's FMA/SFU distribution, but uses a more accurate polynomial
  than the cubic described in its author post. A 100001-point scalar sweep
  measured maximum relative approximation error 3.243e-6 before FP32 rounding.
  Actual GPU output error and latency must determine whether it is useful.

No keys are skipped; FP8 Q/K/V and FP16 probability operands retain the parent's
formats. These are experiments, not promoted changes. The verification kernel
already implements conditional online rescaling, so adding that again would
not be a new optimization.

### Completed first verification sweeps

`verify_pipeline.log` contains 72 valid paired comparisons. At batch 8,
stages=3/splits=32 improves 1.421–1.428x over the current split=16 parent.
At batch 16 the parent remains best. At batch 32 stages=2/splits=8 improves
only 1.021–1.024x; this alone is not enough evidence for an official promotion.

`verify_exp.log` contains 12 valid comparisons. Both FMA polynomial variants
regress badly (roughly 0.47–0.58x), despite output relative errors below 0.001.
Reject them. This experiment does not establish that SFU work is the bottleneck.

Next evidence: `verify_direct.log` removes the up-front shared-memory cache-row
lookup table, loading page-table entries as each tile is issued. This reduces
shared memory per CTA at the cost of repeated global metadata loads. A separate
Nsight run (`verify_ncu.log`) profiles the validated verification kernel rather
than extrapolating from the previously profiled single-token decode kernel.

### Verification Nsight evidence

`verify_ncu.log` completed: 346.66 us profiled kernel, 39.95% DRAM throughput,
39.98% compute throughput, 128 registers/thread, 98.30 KB dynamic shared memory,
25% theoretical and 22.79% achieved occupancy. Both registers and shared memory
limit block residency. Shared-load bank conflicts were 4-way on average;
shared-store conflicts 7.1-way. These diagnostics support testing V staging
layouts rather than assuming exponential throughput is limiting. Nsight's
estimated speedups are suggestions, not measured improvements.

`attention_verify_vswizzle.py` XOR-swizzles 16-byte V chunks by key row in shared
memory and assigns fixed adjacent chunks to each conversion thread. Global
cache layout and WGMMA operand layout stay unchanged. Its paired timing and
page64/scaled/odd-batch reference checks run in `verify_vswizzle.log`.

`verify_direct.log` returned all 36 planned comparisons with passing local
output checks before Modal printed `Runner terminated` during shutdown. At
scored batches 16/32 every direct-address variant regressed (best about 0.82x
and 0.83x respectively). Reject the address-table removal. Batch-8 gains come
from the higher split count and remain below the original-layout split-32 gain.

### V swizzle outcome

All six paired comparisons produced identical outputs; three independent
reference checks passed, including page64 and nonunit KV scales. Batch 8 gained
4.5%; batch 16 gained 0.8–1.1%; batch 32 gained 1.1–1.5%. Keep as an experimental
building block, not a promoted change: scored-shape gains are too small to
justify a private check alone. All launched verification experiments finished.

Next direction: profile the swizzled kernel with source-level attribution and
examine the remaining V conversion/store conflicts and register residency.
The measured layout improvement is small despite large aggregate bank-conflict
counts, so those counts alone do not prove a large attainable speedup.

## Register residency and independent KV-head experiments

- `verify_residency.log`: test launch bounds (256 threads, minimum 1/2/3 CTAs),
  stage counts 2/3 and split counts 16/32/64 on the V-swizzled source, explicitly
  requesting shared-memory carveout 100. Intended to distinguish occupancy
  gains from spill costs.
- `verify_pv64.log`: replace padded 72-column P×V/denominator MMA with a
  64-column MMA and FP32 per-lane denominator accumulation, followed by warp
  reduction. Six paired outputs and all three reference checks passed, but the
  scored batches did not improve. Do not promote.
- `verify_onehead_v2.log`: one 128-thread CTA per KV head rather than two
  warpgroups per CTA; halve raw staging and conversion storage per CTA, retain
  shared precomputed physical rows, and fetch each head's own 64-byte K/V span.
  This tests independent scheduling and synchronization. The first attempt
  (`verify_onehead.log`) failed before remote execution because its generator
  hit a replacement assertion; the generator was corrected before v2.

Residency sweep completed 48 comparisons: forcing a three-block launch bound
regressed to roughly 0.35–0.41x. Relaxing to one block with three stages gained
about 2% at batch 16 and under 1% at batch 32; no promotion warranted.
`verify_residency.json` preserves all paired results.

Independent-head source passed six paired output comparisons exactly and all
three reference checks. It improved batch 8 by 21%, but regressed at batches
16/32 under the inherited split policy. `verify_onehead_tune.log` therefore
sweeps stages 2/3 and splits 4/8/16/32 on the scored batches before discarding
this scheduling change.

A native MoE batch-16 Nsight run is in `moe_ncu_v2.log`. The initial app was
explicitly stopped after discovering an incorrect kernel-name filter, before
using its output as evidence. The corrected filter is `wm_gemm`, two launches.
The driver uses the common native Triton finish (same diagnostic limitation as
other MoE probes); the target gate/up and down kernels are unchanged.

## Follow-up after official one-head success

Official private `5971f63d8a544738` passed all 16 tests/12 cases at 1.604674773x,
versus the previous private 1.573111717x. Promoted via the standard source/hash
and coverage gate. Public job `79b879a926b14de9` is pending; never infer its
score from the private result. The exact WGMMA path is also integrated into a
whole-decoder candidate (`decoder_verify_onehead_s2`, job `6e76d74b52f04826`),
without changing its decode/sliding paths.

The unconfounded MoE register-residency sweep (`moe_residency.log`) finished:
minimum blocks 1/2/3 yielded aggregate paired ratios 0.956795/0.962644/0.981932.
All local reference checks passed, but reject all three on performance. Modal
app `ap-A1HeaDeBXUk4QMHc25hN9v` stopped with zero tasks. No app setup changes
were made after the user said to ignore the app request.

`verify_sliding.log` tests the native two-stage independent-head kernel on the
2048-token causal sliding window. The diagnostic baseline explicitly uses the
native Triton fallback for batches >4 because this image lacks SGLang's norm
operator. Its ratios are NOT gains over the promoted FA3 implementation; use
absolute latency to screen and the official checker for real integration.

Public one-head job `79b879a926b14de9` completed all coverage at 1.613583060x.
Latest refreshed rank: attention 360, overall 331. The decoder integration
passed all coverage at 1.499811x, below 1.516715x; archived without promotion.

Native sliding WGMMA completed 30 comparisons and two reference checks.
Best raw graph latencies: batch1 18.90us, batch16 28.99us, batch32 37.95us.
These do not beat the official promoted path, so reject. No official check
spent on it. `verify_sliding.json` preserves all results.

A focused official check `attention_slide_fa3_nosplit` / `0707c94f18a34994`
sets num_splits=1 only on the sliding FA3 fallback. The pinned primary API at
`serving/sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_attn.py`
explicitly documents this disables split-KV; default zero uses a heuristic.
Full-attention and native paths are unchanged.

## Focused dense algorithm screen

`dev/cublaslt/lt_probe.py` uses cuBLASLt BF16 operands/output with FP32 compute.
It queries at most 32 heuristic algorithms, tests the first 16 for each of four
shapes, and confirms the best on three new seeds with alternating cold-cache
CUDA graph timings. `drv_lt_projection.py` uses the original dense v5 path as
its baseline on these cuBLAS regimes; this is not a claim about prepare's
compressed-weight choices. Scratch is bounded to 8 MiB. Plans retain shape and
algorithm metadata only, and runs always consume current x and weights.

Primary API reference:
https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgogetheuristic

The purpose is to test algorithm selection rather than assume that changing
frameworks yields speed. This probe is diagnostic only, not a submission.

cuBLASLt screening completed successfully: all confirmed outputs passed the
reference. Pooler batch32 showed 1.026–1.041x over the cuBLAS v5 path; QKV16 and
unpooler32 regressed, and pooler128 had no robust gain. Keep the probe as a
research artifact; do not promote the extra machinery for this narrow result.
All local jobs completed before the next GPU probe was launched.

`attention_verify_tile128.py` doubles K/V keys per tile from 64 to 128 while
retaining independent 128-thread KV-head CTAs. It uses m64n128 FP8 QK WGMMA,
16 packed P fragments, eight P×V instructions, a larger converted V tile and
two raw stages. Relaxed launch bounds permit more registers. This halves tile
synchronization frequency but trades occupancy and register demand; no gain
claimed before `verify_tile128.log` completes. It attends to all keys.

FA3 num_splits=1 check completed all coverage but regressed to 1.574246x.
Official sliding verification latency rose from 22.42 to 29.77us at batch16
and 27.51 to 30.39us at batch32. Reject; the heuristic's extra split/combine
launches buy useful parallelism. Full/decode paths stayed approximately stable.

The first 128-key tile attempt failed local output comparison (relative error
0.995) before timing. Inspection found an out-of-bounds temporary V offset
array: four generated offsets were written into two slots. Corrected the
generator and source, and preserved the failed source as
`attention_verify_tile128_bad_offsets.py`. The single rerun is
`verify_tile128_v2.log`; do not use the first run as performance evidence.

Corrected tile128 rerun passed all six output comparisons (relative error
0.00041–0.00050) and all three reference checks. It improved nearby batch8
by 18.6%, but regressed at scored batch16 (0.892–0.896x) and batch32
(0.852–0.855x). Do not promote this launch policy. Larger tiles help some
scheduling regimes but do not automatically improve the scored batches.
All Modal probes launched in this continuation have finished normally.

Next useful experiment: test the 128-key independent-head path at full-context
verification batch1, with split counts 32/64/128, against the current Triton
fallback. This is a distinct scored regime; changing the split policy may
expose a benefit that did not occur at batches16/32. Keep existing large-batch
paths unless official aggregate evidence supports a replacement.

## Single-request verification follow-up

`drv_verify_single128.py` compares the 128-key independent-head kernel against
the currently promoted Triton fallback at batches1/2, three fresh seeds,
splits32/64/128/256. It also checks page64 edges, nonunit KV scales, and a
one-million-token context. It directly calls the native forward only in the
diagnostic driver; no candidate dispatch is promoted based on timing alone.
One bounded Modal job: `verify_single128.log`.

Single-request tile128 sweep completed 24 comparisons and all three tough
reference checks. At batch1, split128 improved 1.164–1.183x; batch2 split64
improved 1.306–1.327x. Other split counts were materially worse, confirming a
scheduling effect. Results: `verify_single128.json`.

`attention_verify_small128` adds the 128-key tile only for full-context
verification at batches1..4 and contexts >=32768; split counts128/64/32 follow
batch size. It preserves the validated 64-key large-batch path and all decode
and sliding paths. Source is generated from immutable validated parents by
`make_verify_small128.py` and archived before the official check.

## Focused MoE weight-prefetch/resource test

Inspection found RING=0's host launcher allocates the default ring-sized shared
buffer despite this device branch using only cross-warp reduction storage.
`moe_preload2.py` sizes that allocation to the maximum reduction footprint and
halves live prefetched weight chunks (four to two) within each scale group.
K accumulation order, FP4 decoding, BF16 boundaries and scales remain unchanged.
This tests register pressure versus memory-level parallelism, not lower precision.

`moe_preload2_v2.log` compares a single native module against the unchanged
validated parent at five batches/two seeds and three independent reference
cases. As with prior MoE diagnostics, both sides use the same native Triton
finish because the diagnostic image lacks SGLang. The initial local generation
attempt stopped at an assertion before any remote GPU execution.

Official `attention_verify_small128` / `2191c82015cf4143` passed all 16 tests and
12 benchmarks, aggregate 1.611416936x versus parent private 1.604674773x.
The changed scored case (verify1 full) improved from 26.51us to 22.86us, about
13.8% lower latency; other shapes moved within ordinary run variance. The
aggregate CI overlaps, so do not overstate the aggregate gain. Three fresh
local seeds and the official changed-case result support the real algorithmic
improvement. Promote through the normal immutable-source/coverage gate and
collect the subsequent public run before reporting a public gain.

MoE preload2 completed ten comparisons and all three reference checks with
identical outputs to the validated parent. Aggregate ratio 1.002506x is not a
robust win. Batch8 regressed on one seed (0.917x); batch16 improved on the other
seed (1.085x). No official check spent.

`moe_preload_adaptive.py` restores original four-chunk staging when the compile-
time token fragment count NCH=1, and uses two chunks only when NCH>=2. It keeps
the corrected shared-memory allocation. `drv_moe_preload_adaptive.py` tests six
fresh/input seeds at batches8/16/32 and prints real local-expert token counts
outside the timed function to test whether token occupancy explains the mixed
result. This is algorithm selection by work size, not an input-value shortcut;
all selected experts, tokens and products are computed in both branches.

Public small128 job `009d713f9a7b4307` passed all coverage at 1.614018068x,
essentially unchanged from previous 1.613583060x at aggregate level. Report
the measured single-request improvement separately from this near-zero public
aggregate difference. The new private best remains 1.611416936x.

Adaptive preload completed 18 comparisons and three reference checks, all
outputs identical to the parent. Batch16 six-seed geometric mean 1.028523x,
range0.992–1.079x. Routing confirms gains on the three seeds with local-expert
counts >8 (16,13,16 tokens respectively): 1.054/1.079/1.048x. The three seeds
with no such expert stayed approximately neutral. Batch8 mean1.004265x and
batch32 mean1.000566x. This supports NCH-dependent register pressure, but overall
mean1.011042x is modest. Preserve as a candidate; no official check spent yet.

## Exact block-diagonal head packing with sparse tensor cores

New architectural hypothesis: two independent KV heads, each serving eight
query heads, can share one 16-row warp MMA. Interleave their K dimensions and
represent the cross-head zeros using 2:4 metadata. The compressed A contains
every real query element; dense B contains every key from both heads. These
are algebraic padding zeros, not low-attention key selection or value pruning.
FP8 query/key representation and FP32 accumulation stay unchanged.

NVIDIA's primary PTX reference documents sparse FP8 m16n8k64 and fragment/
metadata layouts (section Matrix Fragments for sparse mma.m16n8k64). The
first small numerical prototype is `dev/sparse_head/probe_qk.py` with eight
randomized comparisons in `sparse_qk.log`. It validates tensor-core mapping
only; it cannot establish an end-to-end attention speedup. A matching sparse
FP16 P×V packing would be needed for a complete design.

Primary reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-sparse-mma-16864-u8s8

FuTransformer (Oct16, https://arxiv.org/html/2610.23796v1) also emphasizes
fragment-aligned operands and epilogue fusion. Its INT4 KV/weight precision
changes do not transfer to this challenge; do not import those representations.

Sparse packed-head QK prototype passed all eight randomized cases with ZERO
max absolute difference against dense FP32 products of the same FP8 inputs.
This proves exact head isolation and fragment/metadata mapping on H200.
`sparse_pv.log` now checks the matching FP16 P×V primitive before a full kernel.
The expected benefit applies particularly to decode: two 8-row GQA groups can
fill a single 16-row sparse MMA instead of each padding its own 16-row tile.
Verification's 64-row groups already fill dense tiles, so no analogous padding
speedup is assumed there.

The first sparse FP16 P×V prototype failed (relative error0.952). NVIDIA's
metadata diagram shows FP16 m16n8k32 differs from FP8: low16 bits describe rowg,
high16 rowg+8, while selector0 lanes0/1 cover the two K halves. Correct constant
is 0xEEEE4444 in both owners. The failed source is preserved as
`probe_pv_bad_metadata.py`; corrected run is `sparse_pv_v2.log`.

Corrected sparse P×V primitive passed all eight cases with relative error below
4e-7. The exact algebra and metadata mappings are documented in
`research/ptx_sparse/HEAD-PACKING.md`.

`attention_sparse_head_v1.py` is the first complete decode-only integration:
32-key tiles, two cp.async stages, combined-head FP8 sparse QK, FP16 sparse PV,
online softmax, four-warp merge, and the parent's output norm/gate. V is staged
in an expanded-head row order, so ldmatrix transposes directly into the sparse
PV fragment without an extra converted-V shared buffer. K is read with exact
head-pair assembly. New local job `sparse_decode_v1.log` performs small reference
checks before compiling the large baseline, then sweeps three split policies
on each scored decode batch/layer, two seeds each. No official check yet.

Latest authenticated ranks after public small128: attention362, overall332.

Sparse decode v1 passed all three early reference checks (including page64 and
nonunit scales) and all 36 paired comparisons, with output relative error below
0.0003 versus the parent. However, best paired ratios remained 0.684–0.844x on
sliding and 0.570–0.654x on full; reject this implementation for submission.
The correct algebra did not produce a fast pipeline automatically.

`drv_sparse_decode_profile.py` / `sparse_decode_ncu_v1.log` profiles one native
batch16 full kernel to distinguish register spills, shared instruction pressure,
and memory throughput before revising it. Suspected weakness: scalar shared
loads assembling the interleaved FP8 K operand, plus address arithmetic. The
V path already uses vector ldmatrix loads from the expanded-head layout.
Do not state that this suspicion is proven without reading the counters.

A possible follow-up (not yet implemented) is register-prefetch of K on the
future tile, PRMT packing of two-byte head pairs, and vector ldmatrix loads of
the packed current K tile. Ordinary synchronous global loads must be issued
before current QK/PV work, then consumed/stored only after enough independent
work to hide latency. This trades additional live registers against fewer
shared load/address instructions; resource counters should decide whether it
is sensible. An alternate 16-key tile would halve the prefetch register set,
but adds loop/synchronization overhead and requires measurement.

Sparse decode NCU v1 completed: 217.66us under profiling, DRAM35.76%,
compute30.75%, 168 registers/thread, 71.17KB dynamic shared memory, stack1024B,
achieved occupancy16.55%. Local-memory loads/stores mostly miss L1; long
scoreboard stalls are substantial. The profile supports investigating register
spills and live operands rather than assuming tensor-core throughput is limiting.
Shared-load bank conflicts average1.7-way and account for25.3% of wavefronts.

`sparse_residency.log` now compares (a) relaxed launch bounds128,1 and (b) explicit
memory clobbers on sparse MMA asm to constrain load hoisting, against the
promoted kernels. Both use the same exact arithmetic and selected split policies.

Prepared but not launched: `attention_sparse_head_tile16.py`, generated by
`make_tile16.py` and `build_candidate.py --kernel ... --output ... --module ...`.
It narrows the tile to16 keys to reduce score registers and copy-address live
ranges; this adds loop overhead and must be measured. The candidate assembler
now accepts explicit source/output/module arguments for reproducible variants.

Sparse residency experiment completed all24 comparisons with passing local
output agreement. Relaxed launch bounds improved full batch16 from roughly
223us to162us at the same split128 policy, but the promoted parent remained
about132us. Full batch32 improved from~385us to~269us, still above parent220us.
Explicit memory clobbers did not meaningfully improve the original prototype.
No sparse-head variant is ready for official submission. All Modal jobs from
this continuation are terminal. Next available local candidate is the prepared
16-key tile, to test lower live-register demand; it has not yet run on GPU.

## Narrow sparse tile experiment

`sparse_tile16.log` tests the prepared 16-key tile with launch bounds128,4 and
128,1, plus two split counts per scored decode regime, two seeds. The driver
prints actual register/local-byte attributes for each compiled kernel. It
checks a page64 small reference case before compiling/timing the parent.
One bounded Modal job, no official submission.

Prepared only: `moe_adaptive_weight_cg.py` retains adaptive prefetch depth and
uses L2-cached __ldcg for direct streaming weight loads (not activations/scales).
The earlier MoE profile showed near-zero L1 reuse for streamed weight data and
local stack traffic; bypassing L1 could preserve it for the latter. This is a
hypothesis, not a measured improvement. No GPU job launched for this variant.

Narrow-tile experiment completed48 comparisons and early page64 reference.
Actual attributes: capped128 registers/120 local bytes; relaxed187 registers/0
local bytes. Relaxed tile16 improved sliding16 by3.0–5.0% and sliding32 by5.3–5.4%
at their best splits, but remained slower on full attention. Source and data
remain experimental; results are in `sparse_tile16.json`.

`make_prefetch.py` now changes the K pipeline for the16-key tile. Each warp
prefetches the next K tile into registers, computes the current tile, then packs
head pairs with PRMT into a single shared K buffer. Vector ldmatrix replaces
scalar shared K assembly. V retains two cp.async stages. Raw shared storage
falls from32KiB to24KiB plus row metadata. This trades prefetch registers and
permutation work against fewer shared-load/address instructions. It is a
measured-bottleneck hypothesis, not an assumed gain. `sparse_prefetch.log` runs
one variant, prints registers/local bytes, fails early on a reference check,
then compares24 paired configurations if correct. No official check yet.

Register-prefetched K variant completed24 comparisons and early reference.
Actual registers167, local bytes0. Vector packing improved full16 to144–147us
from the scalar variant's~152us, but parent~131us remained faster; full32 stayed
~252–265us versus parent220us. Sliding gains were marginal. No promotion.

`sparse_lazy.log` tests conditional online softmax rescaling (threshold8 log2
units), using the same algebra already validated in the WGMMA verify parent.
It updates the reference maximum only when needed, keeping FP16 probabilities
<=256 and preserving every contribution and final normalization. Two launch
bounds distinguish the effect of the new branch's register demand. This is
not a probability approximation or key-pruning scheme.

For diagnostic compilation, `compile_split.py` moves trailing Torch host
bindings into the C++ translation unit; device source, device compiler flags,
and launch code stay identical. This avoids parsing Torch headers through
nvcc. The lazy driver uses it for the parent module and reports compile time;
all correctness/timing gates remain the same. No app lifecycle changes made.

Conditional-rescale sparse experiment completed48 passing comparisons but
regressed versus ordinary rescaling. Resources: loose168 registers/0 local
bytes, capped160/0. No sparse-head candidate improves enough to promote.
Set this branch aside rather than keep spending checks on it. The only
small local win remains ordinary16-key relaxed sliding16/32 (~3–5%).

The split host/device compiler helper built the unchanged parent FD device
kernels in31.28s; outputs/timing remained normal. It can reduce diagnostic
compilation overhead without changing device arithmetic or app lifecycle.

`cute_large.log` revisits CuTe only at projection batches128/256, pooler and
unpooler, with128×128,128×256,256×128 and128×64 tiles and splitK1/2/4. Earlier
CuTe screens used smaller tiles; larger output tiles can change weight reuse
and grid-wave utilization. Same explicit capture stream, FP32 partials,
BF16 outputs, and post-capture input mutation check as the corrected driver.
One bounded GPU job, no official check until repeatable useful gains exist.

Large CuTe screen completed20 valid configurations and four rejected unsupported
M=256 tile attempts. All valid outputs passed reference checks after input
mutation. None beat cuBLAS: best pooler128 ratio0.976x, pooler2560.924x,
unpooler1280.871x, unpooler2560.806x. Do not promote this family. The runner
printed `Runner terminated` at shutdown after all planned cases were returned.

`moe_weight_cg.log` now runs the prepared adaptive-prefetch plus streaming-weight
L2-cache policy, six seeds per batch8/16/32 and independent reference cases.
The split compiler helper is used for both the unchanged parent and candidate;
actual device source/flags and graph capture are preserved. One GPU job only.

Streaming-weight CG test completed18 comparisons and three reference checks.
All outputs identical. Aggregate1.010278x matches the earlier adaptive prefetch
result1.011042x; no additional cache-policy gain. Batch16 mean1.02724x, other
batches approximately neutral. Keep the simpler adaptive candidate for possible
future integration; no official MoE quota spent. Split compiler measured64.58s
for the parent and63.14s for this candidate, below prior~86s full-Torch nvcc
candidate builds, though those timings are from different runs.

`router_direct_gemv.log` screens the existing FP32-accumulating Triton GEMV
on the router's original FP32 weights, m=1 only, versus the validated hi/lo
matmul path. Ten BN/BK/warp configurations and three seeds, all outputs checked
against the full reference. No lower-precision operands or sparsity shortcut.
This is a low-compilation-cost single-job test, not a broad CUDA sweep.

Direct router GEMV completed30 comparisons/reference checks. BN1/BK1024/eight
warps was repeatably1.288–1.297x over the parent m=1 router path, using original
FP32 weights and FP32 FMA accumulation. Raw medians~9.66us versus12.46us.
The selected candidate adds only an m=1/router branch using this existing
Triton GEMV; all other projection paths and prepare remain unchanged.
Official check: `projection_router_direct_v1`. No public gain claimed yet.

Official router direct check `ee2cc07a65484f18` passed all9 tests and30 timed
cases at1.213370916x (CI1.195170–1.231571), versus previous private1.182878x.
Promote through the immutable-source/coverage gate and submit publicly. This
is an aggregate official improvement; collect the scored job before claiming
its public score or rank changed.

Direct router candidate promoted; public job `3635720da43d4d58` pending.
Official changed-case latency: router decode1 3.98us versus previous6.96us.
Other performance changes include run variance and prepare's existing autotuning;
do not attribute every aggregate difference solely to the changed branch.

`dense_single_gemv.log` follows the occupancy result on m=1 QKV/O only, seven
metadata configurations per layer, three seeds. Both regimes use v5 GEMV in
the latest official prepare selections. All BF16 weights and FP32 accumulation
remain unchanged. Compare before considering any integration; no CUDA compile
or additional official quota is needed for this local screen.

Public direct-router job `3635720da43d4d58` completed all coverage at1.203575186x,
up from the old projection public1.1821x. Latest private1.213370916x retained.

Singleton dense GEMV screen completed42 reference-checked comparisons. QKV
BN1/BK512/eight warps improved1.172–1.185x; O gains were only~1.6%, so leave O
unchanged. `projection_single_gemv_v2` adds the QKV configuration to the validated
direct-router parent and is sent to the official checker before any promotion.
The existing prepare tuner can still select its lossless compressed path if
it beats the new v5 fallback on a given GPU.

Official singleton QKV check `be2868a5c1934c60` completed with correctness
passed at1.194818145x (CI1.176896–1.212740). This is below the retained
private parent1.213370916x, so do not promote or spend a public submission.
The local17–18% gain did not establish an official aggregate improvement.
Full per-load report: experiments/projection_single_gemv_v2/analysis.md.
Production remains projection_router_direct_v1.

### Dense transposed-head attention prototype

Replace the eight-query-row/padded-M layout with K[16keys,64] x
Q[8heads,64]^T using dense FP8 mma.m16n8k32; compute PV as
V^T[64dims,16keys] x P[16keys,8heads] using dense FP16 mma.m16n8k16.
Both MMA output dimensions are fully used. No key pruning, weight/cache
precision change, sparse metadata or duplicated head data.

`dev/transpose_head/probe.py` stages normal K/V using XOR-swizzled shared
memory, checks transposed ldmatrix operands and the four-shuffle score-C to
probability-B permutation. `probe.log` completed all eight tests across
B1/3/16,16/128/256 keys and two seeds: QK exactly matched FP32 reference;
PV relative error<=1.805e-7. Modal app ap-sD4KSEHRNXhh9KVYIGF3yL ended.

Full candidate `attention_transpose_v1.py` uses four warps, two KV heads
and two independent 16-key partitions per iteration, a two-stage 16KB
cp.async K/V ring, ordinary stable online softmax, and a two-partition
CTA merge. Correctness and paired cold-L2 graph comparison is pending
in `dev/transpose_head/decode_v1.log`. No official check spent.

Transposed v1 completed36 paired comparisons plus3 reference cases: all pass.
Batch16/full splits32 improved1.119x across2seeds; batch32/sliding splits16
improved1.099x. Batch1 loses and will retain validated production path.
Other regimes are mixed; full32 best~1.034x, full16 split64~0.994x.
`tune.log` tests five combinations of stages2/3/4 and launch-bounds
minimum blocks1/2/3 in one compiled module. No promotion yet.

Pipeline study `tune.log` completed100 comparisons, all correct. Five
variants all used72 registers/thread,0 local bytes; launch bounds1/2/3
made no useful resource change. Three stages improve full16/splits32
1.2268x and full32/splits16 1.1392x (two seeds); four stages are a little
worse. Sliding16/stages3/splits16 ~1.086x; sliding32/stages2/splits16
~1.111x with some run-order variance. Batch1 bestfourstage~0.996x, so
retain parent batch1 regardless. Modal app ap-Xe7RQfueG6PnGrmo0jgoIy ended.

Next `tiles.log` compares tile16 and tile32 perwarp, stable and conditional
rescaling, allthree-stage. Conditional rescaling updates running max only
when it would rise by>8 log2 units, so intermediate FP16 probabilities
remain<=256; every real key remains included. Three fresh seeds and
scaled/paged nearby-shape reference checks are required. No official quota
spent until selection.

Larger-tile study `tiles.log` completed96 paired comparisons and8 nearby
reference checks, all passing. Resources (registers/local bytes):
t16stable72/0,t16lazy61/0,t32stable82/0,t32lazy81/0. Stable tile16 is
consistently fastest on full decode: full16/split32 1.2235x (min1.2225),
full32/split16 1.1359x (min1.1347), three fresh seeds. Lazy rescaling and
tile32 are rejected. Sliding16/split16 1.0724x and sliding32/split8 1.0957x.

Selected immutable official experiment `attention_transpose_selected`,
job f75e704b61404709, SHA0597abd272a11144e3c028f821602ecf3f055236542ce24575262362148b1fe6.
Uses stable16-key/warp and3stages. Dispatch q_len1,8<=batch<=64,TP8;
batch1/verify retain production. Full splits target512 CTAs, sliding256;
cap chunks8192 for shared-memory bounds on larger contexts. Source generated
by dev/transpose_head/build_selected.py. Official result pending.

Prepared `decoder_transpose_selected.py` using the same selected attention
module, with existing decoder QKV preparation and output projection. No
projection/MoE modifications. Not launched until standalone official result
passes. `profile.log` checks selected nearby8/17/33/64 batches before
profiling full32 with Nsight.

Selected dispatch profile `profile.log` completed nearbybatch8/17/33/64
reference checks (including page64,scale1/64,edges) and Nsight full32.
Kernel177.60us,DRAM76.78%,memory3.77TB/s,L2throughput86.16%,SMcompute55.21%,
72regs,0spills,46.08KB dynamicshared,21.33% achievedoccupancy. Sharedmemory
limits residency to4CTAs/SM. Excess shared wavefronts only~1% of total,
so avoid treating the generic bank-conflict hint as a dominant bottleneck.
Nsight measured17.3M non-fused FP32 instructions; test explicitex2.approx
and fused score scaling/subtraction as targeted instruction reduction.
`ex2.log` compares stable parent,inlinePTXex2,ex2+FMA on3freshseeds, with
selectedfull/sliding split counts, plus nearby referencechecks.

Official selected transposed attention check f75e704b61404709 passed all16
tests and12 timed cases at1.709688113x,CI[1.684042791,1.735333435], beating
parent1.611416936x. Full16 latency122.05->96.81us;full32 207.61->183.14us;
sliding16 13.52->11.89us;sliding32 17.22->14.96us. Other unchanged-case
variation is not attributed to the new kernel. Promoted via coverage/hash
gate; public job07e873299b754f80 pending.

Decoder integration launched as95f251028c7b4043,experimentdecoder_transpose_selected.
No projection or MoE change mixed in.

Explicit exponential study completed36 paired comparisons+6 nearby reference
cases, all passing,72regs/no spills. ex2 improves full~1% but sliding is
mixed; keep separate from the official winner. Directpage-row lookup study
completed24 paired comparisons+4 checks, all passing;71regs/no spills but
full16 only0.796x and full32 0.725x of oldproduction versus selected1.225x/1.142x.
Reject directrow lookup; loading page metadata once and staging rows is
essential here. Both Modal apps ended. Latest currentverify NCU job pending.

Current production verify profile (one independent-KV-head128-thread CTA)
is in dev/transpose_head/verify_current_profile.log:312.45us,DRAM47.23%,
2.32TB/s,L2throughput72.73%,compute42.10%,122regs/thread,41.98KB shared,
23.05% achievedoccupancy. Sharedload/store bankconflicts~34%/~29%; source
summary47% excessive shared wavefronts. This is materially different from
the older256-thread/two-head profile in verify_ncu.log.

Canonical Q/K WGMMA64B swizzle hypothesis: replace interleaved K-major
core matrix physical layout with Swizzle<2,4,3> of row-major64-byte rows,
descriptor mode2,SBO512,LBOignored. Q/K second32-byte fragment now starts
at+32 bytes (descriptor+2), not+256(descriptor+16). Preserve unchanged
PV layout, every-key masks,FP8 Q/K andFP16 P/V. Source mapping follows
NVIDIA PTX canonical K-major64B layout:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout

Generated by dev/verify_swizzle/build.py from official selected parent.
qksw64.log first checks fullverify batch8/page64/edges andbatch9/page64/
scale1/64, then compares3freshseeds atbatch8/16/32. No official quota spent
for this swizzle yet.

Public attention07e873299b754f80 completed1.703190793x,CI[1.677642931,
1.728738655], all coverage passing. Previous public1.614018068x.
Refreshed attention rank314;overall337 (competition also changed).
Top50 remains unachieved.

Decoder95f251028c7b4043 passed all13 tests and12 timed cases at
1.534501759x,CI[1.511484232,1.557519285], versus retained1.516715395x.
Promoted through coverage/hash gate;public15211631065242e0 pending.
Decoder gain is modest and its interval overlaps the parent; do not claim
that every changed timing is causal. Integration changes only attention.

QK-only swizzle completed9 bit-identical paired comparisons+2 reference
checks. Batch8 0.9922x,batch16 1.0163x,batch32 1.0182x. QK+PV canonical
64B swizzle also completed9 bit-identical paired comparisons+2 checks:
1.0141x/1.0114x/1.0179x. No meaningful incremental PV gain. Keep both
experimental; do not spend official quota on these marginal results.
All local Modal jobs in this turn are terminal. Only decoder public result
is pending.

Public decoder15211631065242e0 completed at1.519574061x,CI[1.496780450,
1.542367672], all coverage passing; oldpublic~1.5143x. This small public
difference is within timing uncertainty. Retained source remains the fully
validated transposed decoder; do not claim a strong overall decoder gain.

New architecture prototype dev/transpose_verify:16warps/512threads, one
querytoken/KVhead perwarp, shared K/V raw3stage16-key tiles, one cooperative
FP8->FP16 V conversion reused by all8querytokens. Dense transposed QK and
PV both retain full tensor dimensions; causal masks apply independently
perquerytoken. PART uses64queryrows perKVhead; no interwarp partial reduction
is needed. This trades lower register pressure and V-conversion reuse for
more synchronization. Source generated by build.py; first local validation
is dev/transpose_verify/v1.log. No official quota spent.

Transposed verify v1 completed9 paired comparisons and2 referencecases,
allcorrect, butfull32 only~0.29x ofproduction (~1.15ms versus~335us).
Nsight dev/transpose_verify/profile.log full16:560.13us,64regs/thread,
0localbytes,44.75%occupancy,DRAM12.70%/624GB/s,compute64.03%. Thus
registerspills are not the cause; smaller perwarp querytiles cause too
much instruction/shared-memory traffic. Tile32 derivative prepared but
not tested; do not launch a broad sweep of this weak architecture.

WGMMA overlapv1 (nextQK queued after currentPV) completed9 pairedcomparisons
and2 referencechecks,bitidentical, but~1.00x. Next v2 issues nextQK before
currentVconversion/softmax, then queuescurrentPV, waitsfornextQKonly, and
keepsPVpending until nextiteration before modifyingits registers/sharedV.
Uses separate32-registerscorebuffers,3rawstages and128threadmin2blockbounds.
P operands remain live across asyncuse through explicitcompiler constraints.
NVIDIA PTX requires waiting on bothC accumulators andregisterA operands
before accessing them; source follows this ordering:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group
Sourcegeneratedby dev/verify_overlap/build_v2.py,logv2.log pending.

Overlapv2 completed9 paired comparisons+2referencechecks,allbitidentical,
but~16%slower atfull32. Extra scorebuffer/registerpressure and pipeline
staging outweigh overlap; retain production. Both overlapModalappsended.

Next targeted projection screen usesBF16 tl.dot with BM16/32 instead of
largeM CuTe tiles. QKV splitK4/8/16 andO splitK1 (avoid merge overhead),
BN32/64/128,BK64/128, and normal/transposed BF16 weights. Six targeted
configurations perlayer,2batchsizes,2layouts=48 referencechecked graph
comparisons. One seed first; only clear winners get fresh-seed confirmation.
Driver drv_projection_smalltiles.py,log projection_smalltiles.log.

Projection smalltile screen completed48/48 reference-checked comparisons.
QKV16 best normalW/BM16/BN64/BK128/split8 1.0389x, QKV32 best0.9943x;
O16 best1.0187x,O32 best0.9573x. No clear aggregate winner; preserve
production and do not spend official quota.

MoE outer-loop experiment changes only RING0's `#pragma unroll` on q/GPW
to `#pragma unroll 1`. The old profiled kernel has64registers and local
memory traffic; full unrolling may hoist weights across scalegroups and
increase code size. Inner4-chunk prefetch/dequant/MMA order andprecision
are unchanged. Test5batches x3seeds using paired coldL2 graphs andrelative-L2
comparison, plus reference checks with same diagnostic nativefinish on
bothsides (Modal noSGLang). Complete integration still requires official
worker if any repeatable gain appears. Source moe_outer_loop.py,driver
drv_moe_outer_loop.py,log moe_outer_loop.log.

MoE rolledouterloop completed15 pairedcomparisons+3referencechecks,allpass,
maxrelativeL2versusparent=0.0. Aggregate localratio0.974895x;batch1/8/16
0.9585/0.9473/0.9692x;batch32/128neutral. Reject; fullunrolling is useful
here despite earlier resourcehypothesis. Basecompile74.35s,candidate63.59s.
No official checks launched from this turn's experiments. All localGPUjobs
are terminal. Production hashes match allfour validated entries.
Refreshed overallrank364; top50remains unachieved.

### Native FP8 P/V primitive — corrected baseline precision

Pinned SGLang selects sgl-attn f89bc2306632d1ec5f97b014dded4254f5b4a907.
Its mainloop uses FP8 Element for both P and V; Softmax usesMax_offset8
andFP32 sums. Earlier FP16-only probability assumption was unnecessarily
restrictive; sourceaudit is in research/SEARCH-RESET-2026-10-27.md.

Native primitive first failed an overly strict FP32-comparison threshold
(relativeerror6.012e-5), so we did not assume either correctness orfailure.
Stronger followup: identity/ones/integer/permutation cases areexactzeroerror;
all9randomcases exactlymatch torch._scaled_mm FP8 fastaccumulation outputs.
Maximum errorvsFP32=7.18456358e-05;sameerroroccursin
PyTorch's FP8 path. Thus byte transpose andquadshuffle registerpacking are
validated independently of FP8 hardware arithmetic error. Probeappended.

Fullnativev1 keeps onehead/128threadCTA,FP8probability scaling256,FP32
unquantized probabilitysums,stablemax,exactbyteVtranspose,twoFP8PV
WGMMA operations per64keys (instead offourFP16),smallerVsharedworkspace,
and unchanged all-key masks. Existing decode/smallverifypaths unchanged.
Source dev/fp8_probability/build_native.py. Screen native_v1.log checks
reference oneachcase, not closeness to the olderhigherprecisionFP16Ppath.
No official quota spent until speed and correctness established.

NativeFP8Pv1 completed9timings+11referencechecks,allpassing, butspeedups
0.9478x/0.8844x/0.8975x atbatches8/16/32. Noofficialquota spent.

Independentvectorizedtranspose primitive (exactKorder) passed allfour
structuredtests andnine randomcases exactlyagainst PyTorchFP8GEMM.

Jointlayoutv2 removesallquadshuffles from P packing andVtranspose by
usingthe samewithin16-key permutation onbothoperands:
physical(k)=(k&1)|((k&8)>>2)|((k&6)<<1). It isbijective; fragmentpositions
4t..4t+3 matchlogicalkeys2t,2t+1,8+2t,9+2t. Valuesusetwoldmatrix/
stmatrix passes withfourbyteperms each; noFP16expansion orscalarstores.
Custom per-rowstmatrixaddresses keep outputD ordernormal. This follows
the paired-layout principle also used by pinned FA3; no claim ofacademic
novelty. Our primitive provesexactallfourstructuredcasesandmatches
PyTorchFP8GEMM exactlyacrossnine randomcases. Fullscreenjoint_v2.log
nowpending.

Joint primitive passed4exact structuredcases+9randomPyTorchFP8 comparisons,
allzeroerrorversus PyTorchFP8. Fulljointx4 compilerhitPTXAS12.8segfault
whileNVCC->PTX succeeded. CPU-onlycompiletriageavoids H200allocation.
NeitheroptimizationlevelO2norrelaxed launchboundsfixedit. Movingtranspose
beforeQK, orusingstmatrixx2insteadx4, eachcompilescleanly119regs/0spills;
combiningboth120regs/0spills. Wordstorefallbackalso118regs/0spills.
Thus thefailure is tiedto schedulingx4matrixstores amidoutstandingQK
inthisfullkernel, not a numericalfailure. Keepthesecompilerworkarounds
explicit. CPUcompiler tooling:dev/modal/compile_device.py.

FullGPUcomparejoint_compare.log tests thetwo119-registervariants (overlap
withx2stores;serializedx4stores), referencechecking eachcasebeforepaired
coldL2 graphbenchmarking. OneH200job; CPUcompilejobsareallterminal.
