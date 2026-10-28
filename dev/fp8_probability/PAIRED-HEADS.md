# Cooperative two-head loading

Hypothesis: fetching only one 64-byte KV head per CTA wastes transaction
efficiency on the adjacent two-head cache row. Group two independent 128-thread
WGMMA warpgroups in one 256-thread CTA, and have each warp load complete
128-byte two-head rows. Route each 16-byte copy into the appropriate head's
shared-memory stage. Each warpgroup retains its own QK, FP32 softmax, FP8 PV,
and output accumulator; no keys or heads are skipped.

The two heads have the same sequence/window boundaries, so their CTA-wide
barriers execute the same number of times. Row-index arrays are identical
copies. Shared staging is aligned to 1 KiB for each head. CTA barriers replace
dynamic named per-warpgroup barriers; CPU preflight uses 126 registers, one
barrier, zero stack/spills. This is a performance hypothesis, not a proven
memory bottleneck diagnosis.

Generator: `make_paired_heads.py`; standalone candidate:
`../ptx_campaign/attention_fp8p_paired_heads.py`. H200 correctness and paired
cold-L2 timings are captured in `paired_run.log`. Compare against production,
then against the single-head FP8 parent if a gain survives. No promotion or
official submission before those checks.

Outcome: two early and nine timed task checks passed. All nine paired timing
comparisons lost: batch8 ~0.846x, batch16 ~0.904x, batch32 ~0.834x versus
production. Reject this geometry. Larger transactions did not compensate for
the larger CTA/coupled scheduling; a more specific causal attribution would
require counters and is unnecessary to reject the current implementation.
