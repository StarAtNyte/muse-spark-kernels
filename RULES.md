# Muse Spark challenge rules and organizer clarifications

These notes govern our kernel work and submissions. The [official challenge rules](https://openkernelchallenge.dev/rules#kernels-restrictions), each problem's `TASK.md`, and subsequent organizer guidance are authoritative. Recheck them when guidance changes. The serving track has separate [notes](serving/RULES.md).

## Source and submission

- Submit readable source. Do not embed or load prebuilt binaries, and do not compress, encode, or obfuscate source. Code generation from readable templates is allowed under the published rules. Runtime compilation of readable source is not itself prohibited.
- Respect the official 1 MB submission-file limit; the organizer declined to raise it. Favor maintainable implementations that could plausibly be integrated into SGLang over extensive per-shape tuning overfitted to the challenge.
- All activation-dependent computation belongs in the timed call. Follow the problem contract, use real inputs, and do not manipulate the timing harness, cache or replay outputs, or otherwise game the judge. Check the full rules for other restrictions.
- A passing correctness check is necessary but does not independently establish rules compliance. Audit source, precision and actual selected paths before submitting or promoting. Keep previous results for research, clearly marking candidates with unresolved eligibility.

## Precision and correctness

- For non-router BF16 projections, including K split across CTAs, partial sums must remain FP32 until the final output; do not round them to BF16 before combining them. This is an explicit organizer answer.
- For attention probability P, the organizer agreed that an exp2 approximation (including polynomial or bit-level methods using FP16-rounded scores) and E4M3 storage at a different power-of-two scale from FA3 are allowed within the error bounds **when QK, PV, and all partials remain FP32**. Preserve the problem's other precision and correctness requirements. This answer does not authorize FP16 accumulation of QK or PV.
- Run-to-run routing nondeterminism is allowed when each call is within routing tolerances. The quoted thread did not answer whether occasional tolerance misses are allowed or establish an acceptable miss rate; do not treat a rare failure as permitted. Test routing against the contract and investigate failures.
- The quoted thread did not resolve whether the BF16 projection rule extends to NVFP4 MoE expert partials or answer the separate question about attention partials outside the stated P-approximation conditions. Do not infer permission for lower-precision partials from silence; retain FP32 for the relevant partials unless authoritative guidance permits otherwise.
- Follow all per-problem numerical tolerances on every tested call, including edge cases and held-out or nearby shapes. Do not rely only on the scored shapes.

## Current audit

Some historical scored candidates contain encoded source, and some projection paths use BF16 split reductions. Their published scores are not confirmed eligible. See [the rule audit](research/compliance/2026-11-03-rule-audit.md) and [job inventory](research/compliance/encoded-scored-jobs-2026-11-03.json) before reusing or submitting them. Preserve the experiment history; never submit an encoded or otherwise known noncompliant candidate.

## Organizer clarifications, 2026-11-03/04 (Slack, Justin Wang; supersede the FP32-partials notes above)

- Partial sums (split-K and split-KV, including non-router BF16 projections, attention and NVFP4 MoE) may be held in BF16/FP16, and the combine itself may run in BF16 (e.g. `red.add.bf16x2`), as long as outputs stay within tolerance. The earlier "FP32 until final output" answer is struck.
- The FP32 residual stream may be held in FP16 between two fused steps inside the decode layer, within tolerance.
- GPU fast approximations (fast exp, tanh, rsqrt, etc.) are fine for sigmoid, SiLU and norms within tolerance. A cheaper exp is fine for attention softmax, including attention inside prefill and the decode layer. Where SGLang rounds to BF16 mid-computation, keeping FP32 is fine.
- `prepare` may keep several lossless layouts of the same weights for different shapes.
- Disqualified: using the router's 0.05 score slack to prefer near-tie experts that the batch already loads (gaming the check), and a clamped linear sigmoid. Not every run within tolerance is accepted: degenerate cases or ones that cannot be reviewed are removed.
- Ranking takes the median of reruns, not the best lucky draw. Resubmitting identical code for a better draw does not help.
- Goal reminder: kernels that could plausibly go into SGLang, not extensive per-shape tuning. The 1 MB file limit stays.
- Confirmed on the official rules page (2026-11-05): "Partial sums (split-K, split-KV) can be held in BF16 or FP16 before they're combined, and attention can accumulate QKᵀ and P·V in FP16." Approximate math accurate to BF16 precision is allowed. Final positions are the medians of organizer reruns, including on held-out data; submissions that game the judge are removed, and `okc withdraw <job>` withdraws a run. Serving runs 2026-11-10 to 2026-11-19; serving patches must be reviewable as SGLang PRs.
