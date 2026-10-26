# Muse Spark kernel experiments

Account: `StarAtNyte`. CLI: `~/.okc/bin/okc`.

The first target is `kernels/muse_bf16_proj/submission.py`. Its adjacent
`TASK.md`, `AGENTS.md`, and `reference.py` describe the contract and rules.
`experiments/bf16_starter.py` preserves the original PyTorch starter.

The best validated projection candidate uses singleton dense GEMV, small
pooler split-K, and the permitted BF16 high/low router split. Other dense
projections retain cuBLAS after measured regressions. The best attention
candidate fuses QK normalization, RoPE, FP8 query conversion and cache writes,
uses custom split-KV attention where measured faster, and retains FA3 for
the other regimes. All activation-dependent work is inside the timed call.

## Remote validation

The development machine has an RTX 3050 (6 GB); official checks run on an H200.
The default Python environment has no PyTorch. `.venv` contains Triton 3.6.0
for offline SM90 compilation; this does not provide H200 runtime validation.

First private check: `3f1954adb1044e31`: passed all 9 tests and 30 timed
cases, **1.035754x** over SGLang (95% interval **1.020217–1.051290x**).
The subsequent validated bests are **1.168898x** for projections and
**1.238320x** for attention. Their immutable sources and complete official
results are archived under `experiments/`. `research/best.json` points to the
current best validated implementations, which are copied into `kernels/`.

```bash
~/.okc/bin/okc status 3f1954adb1044e31 --json
python tools/experiment.py collect
python tools/analyze.py experiments/bf16_splitk_v1
python tools/experiment.py check NAME path/to/submission.py --hypothesis 'bounded change'
```

Private checks have a separate daily quota from leaderboard submissions.

| Problem | Best private check | Scored result | Latest scored job |
|---|---:|---:|---|
| Projections | 1.168898x | 1.174299x | 3ee947021eae43b9 |
| Attention | 1.238320x | 1.130524x (prior) | 2c0559f533d145dc (queued) |
| MoE | 1.265151x | 1.146519x (prior) | 33168954e1684bad (queued) |
| Whole decoder layer | 1.097519x | 1.096193x | 4026b7baf4e04652 |

All listed private checks passed complete correctness and timing coverage.
Pending jobs and immutable source snapshots are tracked in `experiments/`.

See [the engineering notebook](research/ENGINEERING.md) for measured
bottlenecks, pending experiments, promotion criteria, and hardware references.
All five challenge contracts are initialized under `kernels/`.

`serving/sglang` is the pinned SGLang checkout. `serving/patch.diff` currently
contains the fused Hopper router and a guarded small-batch expert-reduction
path, both derived from validated kernels. Syntax and patch rules pass; the serving patch has not been
validated on an 8-GPU node. The serving track opens November 5.
