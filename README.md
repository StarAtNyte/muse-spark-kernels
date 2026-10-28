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
the other regimes. Full-context verification uses a joint FP8 probability/value
layout; sliding verification at batch 2–24 uses Hopper cluster shared memory
to fuse attention and its output merge. All activation-dependent work is inside the timed call.

## Remote validation

The development machine has an RTX 3050 (6 GB); official checks run on an H200.
The default Python environment has no PyTorch. `.venv` contains Triton 3.6.0
for offline SM90 compilation; this does not provide H200 runtime validation.
Modal H200 experiments run through `dev/modal/run_h200.py`; see
[the PTX and CuTe campaign](research/PTX-CAMPAIGN.md) for results and caveats.

First private check: `3f1954adb1044e31`: passed all 9 tests and 30 timed
cases, **1.035754x** over SGLang (95% interval **1.020217–1.051290x**).
The current public scores are **1.5346x** for projections, **2.5806x** for
attention, **2.3263x** for MoE, and **2.0746x** for the whole decoder. The
private check scores and immutable sources are archived under `experiments/`;
`research/best.json` points to the current best implementations in `kernels/`.

```bash
~/.okc/bin/okc status 3f1954adb1044e31 --json
python tools/experiment.py collect
python tools/analyze.py experiments/bf16_splitk_v1
python tools/experiment.py check NAME path/to/submission.py --hypothesis 'bounded change'
```

Private checks have a separate daily quota from leaderboard submissions.

| Problem | Public score | Rank | Top-50 cutoff | #1 score | Public job |
|---|---:|---:|---:|---:|---|
| Projections | 1.5346x | #63 | 1.5448x | 1.6367x | d2dd2aec03f94dbf |
| Attention | 2.5806x | #59 | 2.5882x | 3.0316x | 0d181df6ffc2480b |
| MoE | 2.3263x | #68 | 2.3381x | 2.6853x | a834a3c9bf6e44f6 |
| Whole decoder layer | 2.0746x | #93 | 2.1263x | 2.3986x | 2a5a87f725fd47d2 |
| Prefill layer | — | Unranked | 1.7831x | 2.0252x | No submission |

Overall rank is **#93**. Prefill is a separate board and does not count toward
overall rank; we have not submitted a prefill run. Its current top-50 cutoff is
1.7831x. Leaderboard snapshot refreshed 2026-10-28; ranks and cutoffs can move
as new runs are scored. The latest stacked M32/M128
operatorstack transfer passed correctness but scored only **1.8533x**, so it
was rejected and not submitted. See `research/ENGINEERING.md` and
`experiments/decoder_vjeux_operatorstack_stack_v1/` for the score and profile.

## Score progression

Each progression shows published milestones. The rank column gives today's
rank, except where an earlier rank is explicitly called out. Intermediate
ranks were not returned in the public job records, so they are left blank.

| Problem | Score progression | Main technique | Current file | Current rank |
|---|---|---|---|---:|
| Projections | 1.1743x → 1.5177x → **1.5346x** | Adopt Nathen v18's shape-specific BF16 kernels, then stack measured M256-pooler and M128-unpooler cuBLASLt plans. The last run moved to #55 when scored; live rank is #63. Job `d2dd2aec03f94dbf`. | `kernels/muse_bf16_proj/submission.py` | #63 |
| Attention | 1.1305x → **2.5806x** | Replace the initial fused-I/O route with the validated vjeux full attention implementation. Job `0d181df6ffc2480b`. | `kernels/muse_attention_decode/submission.py` | #59 |
| MoE | 1.1465x → **2.3263x** | Replace the small-reduction patch with vjeux's persistent WGS routed-expert kernel and shape dispatch. Job `a834a3c9bf6e44f6`. | `kernels/muse_moe_w4a16/submission.py` | #68 |
| Whole decoder | 1.0962x → 1.7469x → **2.0746x** | Stack M32/M128 expert routes with four-split full verification (1.7469x, then #253); replace that base with the validated vjeux whole-decoder path. Job `2a5a87f725fd47d2`. | `kernels/muse_decode_layer/submission.py` | #93 |
| Prefill layer | No scored run | Not started; separate leaderboard, excluded from overall rank. Top-50 cutoff shown above. | `kernels/muse_prefill_layer/submission.py` | Unranked |

This snapshot was refreshed 2026-10-28. Ranks can move while scores stay the
same; each job ID and score identifies the exact published run.

All listed private checks passed complete correctness and timing coverage.
Pending jobs and immutable source snapshots are tracked in `experiments/`.

See [the engineering notebook](research/ENGINEERING.md) for measured
bottlenecks, pending experiments, promotion criteria, and hardware references.
All five challenge contracts are initialized under `kernels/`.

`serving/sglang` is the pinned SGLang checkout. `serving/patch.diff` currently
contains the fused Hopper router and a guarded small-batch expert-reduction
path, both derived from validated kernels. Syntax and patch rules pass; the serving patch has not been
validated on an 8-GPU node. The serving track opens November 5.

Attention backend: attributed MIT implementation from [team bigboateng](https://github.com/operatorstack/open-kernel-challenge-solutions), independently validated at 1.786690x; a shape-dispatched hybrid with our native/cluster attention is under official evaluation.
