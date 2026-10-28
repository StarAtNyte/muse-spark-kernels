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
The current public scores are **1.5395x** for projections, **2.5806x** for
attention, **2.3290x** for MoE, and **2.0746x** for the whole decoder. The
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
| Projections | 1.5395x | #61 | 1.5455x | 1.6367x | beb3e26afb2544c9 |
| Attention | 2.5806x | #60 | 2.5883x | 3.0316x | 0d181df6ffc2480b |
| MoE | 2.3290x | #63 | 2.3400x | 2.6853x | 88e09ad35c814c29 |
| Whole decoder layer | 2.0746x | #94 | 2.1351x | 2.3986x | 2a5a87f725fd47d2 |
| Prefill layer | 1.7874x | #49 | 1.7847x | 2.0252x | 68ac9af85a3a4e3d |

Overall rank is **#95**. Prefill is a separate board and does not count toward
overall rank. The scored prefill run passed at 1.7874x; its leaderboard rank
is #49. Its current top-50 cutoff is 1.7847x. Leaderboard snapshot refreshed 2026-10-28; ranks and cutoffs can move
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
| Projections | 1.1743x → 1.5177x → 1.5346x → **1.5395x** | Adopt Nathen v18's shape-specific BF16 kernels, then stack measured M256-pooler and M128-unpooler cuBLASLt plans. Then add the paired vjeux M32 pooler and M128 router routes. Job `beb3e26afb2544c9`. | `kernels/muse_bf16_proj/submission.py` | #61 |
| Attention | 1.1305x → **2.5806x** | Replace the initial fused-I/O route with the validated vjeux full attention implementation. Job `0d181df6ffc2480b`. | `kernels/muse_attention_decode/submission.py` | #60 |
| MoE | 1.1465x → 2.3263x → **2.3290x** | Replace the small-reduction patch with vjeux's persistent WGS routed-expert kernel; set activation gather batch to 7 after paired H200 screening and a passing full check. Job `88e09ad35c814c29`. | `kernels/muse_moe_w4a16/submission.py` | #63 |
| Whole decoder | 1.0962x → 1.7469x → **2.0746x** | Stack M32/M128 expert routes with four-split full verification (1.7469x, then #253); replace that base with the validated vjeux whole-decoder path. Job `2a5a87f725fd47d2`. | `kernels/muse_decode_layer/submission.py` | #94 |
| Prefill layer | **1.7874x** | Validated vjeux prefill implementation; public job `68ac9af85a3a4e3d`. Separate leaderboard. | `kernels/muse_prefill_layer/submission.py` | #49 |

This snapshot was refreshed later on 2026-10-28. Ranks can move while scores stay the
same; each job ID and score identifies the exact published run. The latest MoE
submission improved the live rank from #71 to #63; its 2.3290x score remains
below the current top-50 cutoff of 2.3400x.

### Latest validation against the published kernels (2026-10-28)

Three fresh official H200 checks passed full correctness coverage, but each used
the exact same source hash already published for that problem. Their different
timings are run-to-run variation, so they are not new score milestones and were
not resubmitted:

| Problem | Check score (95% interval) | Source vs published | Decision |
|---|---:|---|---|
| Attention | 2.57313x (2.53453–2.62387) | Identical | Keep published 2.5806x |
| MoE | 2.32608x (2.29119–2.36097) | Identical | Historical check; superseded by GATHB=7 run 2.3290x |
| Whole decoder | 2.09091x (2.05955–2.12228) | Identical | Keep published 2.0746x; interval overlaps |

Recent bounded H200 screens also rejected MoE gather batch sizes 4, 6, and 16;
attention full-decode split counts 4–32, alternate CTA grids, and DSMEM cluster
reduction. The screen logs, exact source snapshots, and official check artifacts
remain under `experiments/`; details and next hypotheses are in
`research/ENGINEERING.md` and `research/PTX-CAMPAIGN.md`.

Before the current prefill submission, two private H200 checks of Marlin expert
tile-size heuristics passed correctness but scored **0.9762x** and **0.9899x**;
both were rejected. BM32/BM48 hurt the high-weight single-request 753-token
sliding case, so those candidates were rejected. Full artifacts: `experiments/prefill_blocksize_density_official_v1/`
and `experiments/prefill_blocksize_48_t753_check_v1/`.

All listed private checks passed complete correctness and timing coverage.
Pending jobs and immutable source snapshots are tracked in `experiments/`.

See [the engineering notebook](research/ENGINEERING.md) for measured
bottlenecks, pending experiments, promotion criteria, and hardware references.
All five challenge contracts are initialized under `kernels/`.

Latest decoder transfer check `04a138c2ac314122` passed correctness at
2.08635x (95% CI 2.05506–2.11765) after changing only the WGS activation
gather batch from 8 to 7. It did not beat the prior same-source check
(2.09091x), so it was archived without submission. The live board remains
overall #94; continue targeting integrated decoder dataflow and the category
cutoffs above.

`serving/sglang` is the pinned SGLang checkout. `serving/patch.diff` currently
contains the fused Hopper router and a guarded small-batch expert-reduction
path, both derived from validated kernels. Syntax and patch rules pass; the serving patch has not been
validated on an 8-GPU node. The serving track opens November 5.

Attention backend: attributed MIT implementation from [team bigboateng](https://github.com/operatorstack/open-kernel-challenge-solutions), independently validated at 1.786690x; a shape-dispatched hybrid with our native/cluster attention is under official evaluation.

Latest prefill progress: the vjeux public candidate passed the full H200 check
at **1.79783x** (95% CI 1.77086–1.82479), and was submitted as
`68ac9af85a3a4e3d`, scoring **1.78740x** (95% CI 1.76059–1.81421); its live rank is #49. The exact candidate
and check profile are in `experiments/prefill_vjeux_public_check_v1/`.

Prefill four-split attention passed the official check and a paired H200 component screen; scored submission `69464bf8cbce4588` passed at 1.78568x; the public best remains 1.7874x.

Projection hybrid with vjeux M32 pooler and M128 router passed full check at 1.52701x after paired H200 screens; scored job `beb3e26afb2544c9` passed at 1.53951x, rank #61.
