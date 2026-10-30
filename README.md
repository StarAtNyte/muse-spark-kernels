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
The current public scores are **1.5395x** for projections, **2.6177x** for
attention, **2.3820x** for MoE, and **2.2603x** for the whole decoder. The
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
| Projections | 1.5395x | #138 | 1.5638x | 1.6498x | beb3e26afb2544c9 |
| Attention | 2.6177x | #93 | 2.6681x | 3.2221x | b95cd80bd3e6497f |
| MoE | 2.3820x | #76 | 2.4158x | 2.7578x | 190d8914bacb4aa2 |
| Whole decoder layer | 2.2603x | #28 | 2.2239x | 2.4178x | b72f02c7eb8745f5 |
| Prefill layer | 1.7874x | #122 | 1.8065x | 2.0325x | 68ac9af85a3a4e3d |

Overall rank is **#28**. The table and overall rank were fetched from the
leaderboard on **2026-10-30 at 09:21 UTC**; ranks and cutoffs move as others
submit. Prefill is a separate board and does not count toward overall rank.
The decoder previously reached #25 in an earlier snapshot.
The latest stacked M32/M128
operatorstack transfer passed correctness but scored only **1.8533x**, so it
was rejected and not submitted. See `research/ENGINEERING.md` and
`experiments/decoder_vjeux_operatorstack_stack_v1/` for the score and profile.

## Score progression

Each progression shows published milestones. The rank column gives today's
rank, except where an earlier rank is explicitly called out. Intermediate
ranks were not returned in the public job records, so they are left blank.

| Problem | Score progression | Main technique | Current file | Current rank |
|---|---|---|---|---:|
| Projections | 1.1743x → 1.5177x → 1.5346x → **1.5395x** | Adopt Nathen v18's shape-specific BF16 kernels, then stack measured M256-pooler and M128-unpooler cuBLASLt plans. Then add the paired vjeux M32 pooler and M128 router routes. Job `beb3e26afb2544c9`. | `kernels/muse_bf16_proj/submission.py` | #82 |
| Attention | 1.1305x → 2.5806x → **2.6177x** | Replace the initial fused-I/O route with the validated vjeux full attention implementation (job `0d181df6ffc2480b`), then tune the 16-request sliding decode plan (8 splits, no cooperative finalize). Job `b95cd80bd3e6497f`. | `kernels/muse_attention_decode/submission.py` | #89 |
| MoE | 1.1465x → 2.3263x → **2.3290x** | Replace the small-reduction patch with vjeux's persistent WGS routed-expert kernel; set activation gather batch to 7 after paired H200 screening and a passing full check. Job `88e09ad35c814c29`. | `kernels/muse_moe_w4a16/submission.py` | #96 |
| Whole decoder | 1.0962x → 1.7469x → 2.0746x → 2.0874x → **2.0926x** | Stack M32/M128 expert routes with four-split full verification (1.7469x, then #253); replace that base with the validated vjeux whole-decoder path, then transfer small output projections and gather batch 7. Job `6cf1be8d60e84ac7`. | `kernels/muse_decode_layer/submission.py` | #110 |
| Prefill layer | **1.7874x** | Validated vjeux prefill implementation; public job `68ac9af85a3a4e3d`. Separate leaderboard. | `kernels/muse_prefill_layer/submission.py` | #74 |

This snapshot was refreshed later on 2026-10-28. Ranks can move while scores stay the
same; each job ID and score identifies the exact published run. The latest MoE
submission improved the live rank from #71 to #63; its 2.3290x score remains
below the current top-50 cutoff of 2.3519x.

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

See [H200 lower bounds and bottlenecks](research/H200-LOWER-BOUNDS.md) for per-case judge floors, the underlying models, and measured comparators.

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
`68ac9af85a3a4e3d`, scoring **1.78740x** (95% CI 1.76059–1.81421); its live rank is #74. The exact candidate
and check profile are in `experiments/prefill_vjeux_public_check_v1/`.

Prefill four-split attention passed the official check and a paired H200 component screen; scored submission `69464bf8cbce4588` passed at 1.78568x; the public best remains 1.7874x.

Projection hybrid with vjeux M32 pooler and M128 router passed full check at 1.52701x after paired H200 screens; scored job `beb3e26afb2544c9` passed at 1.53951x, then rank #61.

Projection four-route hybrid adds paired-winning M1 unpooler and M256 router routes; check `4759f26857a74c46` passed at 1.52412x and scored job `4ed546f6ab2748cb` passed at 1.51826x; public best remains 1.53951x.

The isolated vjeux M256 router candidate passed check `593651829e0e4b78`; scored job `df44909a5b3540ff` passed at 1.52297x; public projection best remains 1.53951x.

Decoder T=1/8 output-projection transfer passed check `7ec7babbe7044aa6` at 2.08690x; scored job `1c4e4524414c407f` passed at 2.08743x, rank #95.

Decoder projection and MoE gather stack passed check `9ace173c90844588` at 2.08859x; scored job `6cf1be8d60e84ac7` passed at 2.09262x, then rank #85.

MoE shape-selected natural unit order passed check `72c6e933f69b4bad` at 2.32986x; scored job `6ae96ed05c124b5f` passed at 2.32168x; published MoE best remains 2.32905x.

Prefill boundary experiment: a first-recompute boundary launch adjustment passed full check `17ded4dec47047ad` at 1.79396×; three paired component repeats showed a 0.9–1.4% boundary improvement, too small to establish a full-score gain. The exact checked file scored 1.76757× as `9937a79b7f8448c4`, below the 1.78740× published best; intervals overlap and the incumbent remains unchanged. Live prefill rank #114, #50 1.8046×, leader 2.0325× (2026-10-30). Source, timing logs and uncertainty are in `experiments/prefill_boundary_geometry_v1/`.

KPERM3 progress (2026-10-30): our GATHB=7 MoE stack passed official check `2ecdb5bf891d40e4` at **2.39112×** and was submitted as `190d8914bacb4aa2`; the integrated alygg77 decoder passed check `13161eab55cb447a` at **2.20175×** and was submitted as `78df8889fbbe4803`. The scored jobs passed: **2.3820× MoE (#70)** and **2.1957× decoder (#78)**; overall rank **#78** at the 2026-10-30 snapshot. MoE #50 was 2.4150×; decoder #50 was 2.2189×. The lossless K rearrangement comes from [alygg77's Apache-2.0 kernels](https://github.com/alygg77/okc-muse-spark-kernels); source, notices, and check profiles are archived in `experiments/`.

Decoder stacking: the combined compact T128 QKV and O projection routes on the KPERM3 decoder passed check `851e490e09f44621` at 2.19920×; its profile confirms both kernels ran. Scored job `b72f02c7eb8745f5` passed at **2.26032×** (95% CI 2.22642–2.29423), placing #25 at this snapshot; the point estimate is above the parent’s 2.19567×. Separate-run noise means the entire difference cannot be attributed solely to these routes. Details and the failed preparation attempt are in `experiments/decoder_alygg_bigproj_screen_v1/`.
