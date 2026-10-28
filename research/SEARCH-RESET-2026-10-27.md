# Search reset after stalled kernel experiments — 2026-10-27

The goal remains a top leaderboard result, initially top 50 overall. Current
validated improvements do not establish that goal. This note changes the search
method, not the definition of success.

## What went wrong in this campaign

1. **We audited the baseline too late.** We assumed attention probabilities had
   to remain FP16 and designed expensive FP8-V-to-FP16 conversion paths. The
   exact pinned SGLang FA3 implementation instead uses its FP8 `Element` for
   P and V, FP32 softmax sums, and an exponent offset of eight. This invalidates
   a central optimization constraint we imposed on ourselves. It is evidence
   about the baseline precision; a new implementation must still satisfy the
   task's precision rules and every-key requirement.
2. **Too much local engineering inside one design family.** Most late register,
   swizzle, stage-count, and prefetch changes moved performance by 0–3% or
   regressed. Several would have negligible effect on the weighted track score
   even if their local measurements were repeatable.
3. **Profiles were sometimes used as prescriptions rather than hypotheses.**
   Higher occupancy did not make the transposed verify kernel faster; it had
   no spills and ~45% occupancy but only ~0.62 TB/s. Generic bank-conflict hints
   did not predict the benefit of changing Q/K or P/V layouts. Counter changes
   must be connected to runtime through a small discriminating experiment.
4. **Research was not translated into explicit design constraints soon enough.**
   We had read papers and implemented checks, but kept making proposals within
   an increasingly stale architecture. More code and more iterations were not
   fixing the missing design insight.

What was already sound: immutable submissions, complete official coverage,
paired cold-L2 graph timing, fresh seeds, preserving validated bests, and
avoiding public submissions for clear local regressions. Keep those mechanisms.

## Relevant primary research and engineering sources

| Source | Finding relevant to us | Change to our workflow |
|---|---|---|
| [D2K-Bench, October 2](https://arxiv.org/abs/2610.03226) | Separates algorithmic insight, dataflow, and low-level tricks. Expert guidance materially improves results; runtime alone hides design-discovery failures. | Write and verify the design before choosing PTX/CuTe/Triton syntax. |
| [KernelOPT, revised October 6](https://arxiv.org/abs/2609.30059) | Preserves vendor dispatch and verifies optimizations at model scope; overall gains are much smaller than optimized-only gains. | Beat cuBLAS/FA3 on the actual workload before replacing them; measure integration. |
| [KernelPro, revised August 8](https://arxiv.org/abs/2606.26453) | Uses bottleneck classification plus micro-profiling and search memory, rather than passing raw counters directly into code generation. | Require a falsifiable bottleneck hypothesis and one discriminating test. |
| [Atrex, July 16](https://arxiv.org/abs/2607.14541) | Trace-derived importance weights, production baselines, and optimization dropout to leave stalled search contexts. | Use scored-load impact; reset a failed architecture instead of endlessly mutating it. |
| [KernelBrain](https://arxiv.org/abs/2608.02611) | Cheap candidate screening, then higher-fidelity evaluation for promising survivors. | Small early screen; fresh-seed paired confirmation; official evaluation last. |
| [PyTorch KernelAgent, March 6](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | Hardware-guided design exploration and benchmark isolation. Its published comparison uses default torch.compile with CUDA graphs disabled. | Keep hardware feedback and exclusive GPU measurement; do not transfer headline speedups to our stronger cuBLAS/FA3 baseline. |
| [CloudRift/Emmy, August 1](https://riftstack.ai/research/optimizing-gemma-4-12b-rtx) | Distinguishes per-kernel wins from end-to-end gains and searches structured schedule moves. Hardware is RTX 5090, not H200. | Reuse the methodology, not architecture-specific numbers or unsupported precision choices. |
| [GPU MODE reference kernels](https://github.com/gpu-mode/reference-kernels), [lectures](https://github.com/gpu-mode/lectures) | Executable problem sets, reproducible evaluators, architecture-specific examples. | Prefer code and measured traces to social-post performance claims. |
| [KernelFoundry](https://github.com/isl-org/kernelfoundry) | Hardware-aware optimization targeting multiple languages, with an Intel GPU focus. | It is a search framework, not an automatic H200 solution. |

X/Twitter searches did not yield sufficiently verifiable primary technical
threads in this review. No unverified tweet is used as implementation evidence.

## Exact baseline precision evidence

The pinned SGLang CMake file selects `sgl-attn` commit
`f89bc2306632d1ec5f97b014dded4254f5b4a907`.

- [Pinned mainloop](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp):
  QK and PV both select their operand types from `Element`; the FP8 path
  converts softmax results into that same type before PV.
- [Pinned kernel](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_fwd_kernel_sm90.h):
  FP8 uses `Softmax` with `Max_offset=8`.
- [Pinned softmax](https://github.com/sgl-project/sgl-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/softmax.h):
  sums are computed from FP32 probabilities and corrected for the exponent
  offset before forming LSE.

Copies with upstream copyright headers are in `research/fa3/` for inspection.
No baseline code or verifier was changed.

## Execution rules going forward

- First record exact shapes, legal dtypes, library dispatch, data layout,
  required fusion, and the predicted reduction in scored time.
- `python tools/prioritize.py` reconstructs official scores before estimating
  the impact of a one-shape change. Its projections are within a problem;
  they do not predict overall leaderboard ranks.
- Prioritize approaches with meaningful aggregate headroom. A marginal local
  win is not enough; require a structural reason to revisit a rejected family.
- Separate algorithm/dataflow discovery from low-level tuning. Compare a small
  number of substantially different designs, then tune the winner.
- Use SOL to classify a bottleneck, then isolate it. Do not assume higher
  occupancy, fewer bank conflicts, or lower register counts imply faster code.
- A candidate first passes representative reference checks; an early performance
  screen then decides whether to spend three-seed, full-shape evaluation.
- Preserve immutable winners, all-key computation, fresh-input behavior,
  graph timing, and the full official validation/promotion gate.

## First experiment from the corrected baseline model

`dev/fp8_probability/build.py` produces a Triton verification path with FP8
Q/K/P/V matrix products, FP32 accumulation and softmax sums, stable online max,
and the same exponent offset eight used by pinned FA3. Cache values remain FP8;
external activations remain BF16. Every real key is included, with per-token
causal/window masks. The timing includes cache preparation and final norm/gate.

`dev/modal/drv_fp8p_screen.py` checks paged/scaled/edge cases first, then screens
full-context verify batches 16 and 32. This is a hypothesis, not a claimed win.
Log: `dev/fp8_probability/triton_screen.log`. No official check is launched
until correctness and material performance are established.

The Triton screen completed all representative reference checks and twelve
full-verify comparisons. Best ratios were ~0.505x atbatch16 and~0.497x at
batch32, so it is rejected for performance. This validates the precision
model on those cases, not the dataflow. No official quota was used.

The next original design isolates a native WGMMA FP8 PV primitive: derive
C-fragment-to-FP8-A register packing using quad shuffles and use ldmatrix
plus byte permutation for an exact FP8 value transpose. This removes the
FP16 expansion and reduces PV instruction count. First prove the fragment
mapping numerically before integrating into a coupled-head paged pipeline.
Source:dev/fp8_probability/probe_pv.py; log:native_probe.log.
