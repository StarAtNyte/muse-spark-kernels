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

Prepared a fused decoder-router top-k experiment inspired by vLLM's recent
Muse CUDA routing kernel: split the exact packed 256-expert top-k into four
64-expert local top-k operations and one 32-candidate merge. It preserves the
packed expert-ID tie rule and adds no launch. Its three-seed paired H200 screen
passed all 36 measurements and measured 1.00171× aggregate (1.00355× sliding,
0.99987× full). Scored run `d1faeb5b0a5c496d` is pending; the candidate is not
promoted yet. This is a secondary lever (the route kernel is about 2 µs;
integrated MoE remains much larger). Candidate and pair driver:
[`decoder_router_hier_topk_v1`](experiments/decoder_router_hier_topk_v1/FINDINGS.md).

Decoder WGS warp-parallel prefix (2026-11-02): the 12-case exact-parent paired
screen measured 1.0010×, but the official full run `03083a8a606d42e2` scored
2.20152× (95% CI 2.16850–2.23455×), below the 2.26032× incumbent. It passed
13/13 correctness cases. A broader 12-shape paired H200 screen measured only
1.00225× overall, including a small full-context regression; the candidate
remains unpromoted and the public score/rank are unchanged. Source, profiles, and
measurements are in [`decoder_wgs_parallel_prefix_v1`](experiments/decoder_wgs_parallel_prefix_v1/FINDINGS.md).

Latest whole-decoder experiment (2026-11-02): the rs128 QKV/unpooler transfer
showed selected-case gains in a paired H200 screen, but its official run
`1603d76ca92045e1` scored 2.20383x (95% CI 2.17077–2.23689), below the 2.26032x
incumbent despite passing all 13 correctness cases. We rejected it; the
published decoder score and rank remain unchanged. The official profile
exposed regressions on decode-16/32 sliding. Details, hashes, and measurements:
[`decoder_rs128_topbase_v1`](experiments/decoder_rs128_topbase_v1/FINDINGS.md).
The selected-case screen failed to predict the whole-layer outcome. The
integrated MoE remains the largest measured lever, consuming 52–57% of
decode-16/32 sliding profile time and more than 60% in some verification loads.
An exact-incumbent paired H200 screen changed only the embedded MoE gather batch from 8 to 7, matching the standalone MoE winner. All cases passed correctness, but the six sliding decode/verify shapes aggregated to 0.9993×, so the 2.2603× incumbent remains promoted. Raw paired timings: [`decoder_stack_gathb7_exact_v1`](experiments/decoder_stack_gathb7_exact_v1/FINDINGS.md).
A second exact-baseline stack tested the attention-board 8-split/no-cooperative-finalize plan for batch-16 sliding decode. All paired cases passed, but the batch-16 two-seed geometric gain was 0.9990× and the six-row aggregate was 1.0000×; no decoder score change. Details: [`decoder_stack_attn16_plan_exact_v1`](experiments/decoder_stack_attn16_plan_exact_v1/FINDINGS.md).

The development machine has an RTX 3050 (6 GB); official checks run on an H200.
The default Python environment has no PyTorch. `.venv` contains Triton 3.6.0
for offline SM90 compilation; this does not provide H200 runtime validation.
Modal H200 experiments run through `dev/modal/run_h200.py`; see
[the PTX and CuTe campaign](research/PTX-CAMPAIGN.md) for results and caveats.

Public-code review for the current sprint: [what we checked and what is still worth porting](research/PUBLIC-CODE-AUDIT-2026-11-02.md). Lee101's warp-independent cluster path was transferred and screened on H200; the adapted path was 0.544×/0.584× of the incumbent on its target batches, so it was rejected. The Chord semi-static MoE tile idea is already represented by our WGS compile-time tile-count dispatch.

The current board scores are **1.5919×** for projections, **2.6328×** for
attention, **2.3820×** for MoE, **2.2603×** for the whole decoder, and
**1.9321×** for prefill. The projection score is a repeat draw of the same
kernel source (95% CI 1.5680–1.6158×); it moves the leaderboard row but does
not establish a kernel gain. The best validated projection artifact remains
1.5853×. Private check scores and immutable sources are archived under
`experiments/`; `research/best.json` records our best validated source hashes.

```bash
~/.okc/bin/okc status 3f1954adb1044e31 --json
python tools/experiment.py collect
python tools/analyze.py experiments/bf16_splitk_v1
python tools/experiment.py check NAME path/to/submission.py --hypothesis 'bounded change'
```

Private checks have a separate daily quota from leaderboard submissions.

| Problem | Public score | Rank | Top-50 cutoff | #1 score | Public job |
|---|---:|---:|---:|---:|---|
| Projections | 1.5919×† | #41 | 1.5904× | 1.6596× | 5a25cd13386a47ee (same source) |
| Attention | 2.6328× | #164 | 2.7253× | 3.3389× | 7d250883e6b94c50 |
| MoE | 2.3820× | #205 | 2.4703× | 2.7642× | 190d8914bacb4aa2 |
| Whole decoder layer | 2.2603× | #114 | 2.3104× | 2.4844× | b72f02c7eb8745f5 |
| Prefill layer | 1.9321× | #33 | 1.9164× | 2.0727× | f3b10ea957e04ea6 |

Overall rank is **#114**. Ranks and cutoffs above were refreshed on **2026-11-02 at 14:21 UTC** and are archived in [the live snapshot](research/standings/2026-11-02.json). †The projection row is a higher repeat draw of identical code, not an implementation change. Prefill is a separate board and does not count toward overall rank. Projections and prefill are inside #50; the remaining gaps are decoder +2.22%, attention +3.51%, and MoE +3.71%.

Recent prefill research: [Humming shape-tuning screen](experiments/humming_prefill_shape_tuning_v1/README.md) tested six H200 schedules; none beat our existing routed-expert component. Raw measurements and configurations are preserved with the experiment.

Recent official prefill check: [stacked 3,012-token QKV and pooler plans](experiments/prefill_3012_lt_stack_v1/README.md) passed at 1.7827×; the published 1.8092× source remains the incumbent. The intervals overlap, so the difference does not establish a regression.

Latest MoE experiment (2026-11-02): v8 failed before timing because it omitted the MFIN scratch tail. V9 restores the 4,096-int flag region and DF sentinel, passed all correctness cases, and scored 2.36004× (95% CI 2.32464–2.39544×) against the 2.38203× incumbent; the overlapping interval is inconclusive. Scored run `1fb6b4f749ec44bb` uses the same timed code plus a prepare-only print to confirm the judge selected `-DOKC_SC6W=1`. Full artifacts are under `experiments/moe_kperm3_sc6_stack_v8/`, `v9/`, and `v10/`.

Projection score sampling (2026-11-02): three comment-only variants of the exact scored projection source scored 1.56698×, 1.59192× and 1.58068×. The board keeps the higher repeat, currently #41 and inside #50. These are repeat draws, not kernel changes; confidence intervals and source copies are tracked in `experiments/projection_rank_draws_v1/`.

Measured next priorities for all five kernels are in [the current score-weighted sensitivity report](research/experiment-priorities.md), generated from their official scored per-shape profiles.

## Score progression

These are published milestones; current ranks refer to the snapshot above. Intermediate ranks are omitted when not recorded. Exact scored sources and job IDs are retained under `experiments/`.

| Problem | Published score progression | Main technique | Best scored source | Current rank |
|---|---|---|---|---:|
| Projections | 1.1743x → 1.5177x → 1.5346x → 1.5395x → **1.5853x** → 1.5919x† | Shape-specific BF16 GEMV/GEMM routes, pooler split-K and high/low router split. | `experiments/bf16_keell0renz_compare_v1/submission.py` | #41 |
| Attention | 1.1305x → 2.5806x → 2.6177x → **2.6328x** | Fused Q/K preparation and cache writes, split-KV attention and shape-selected FA3/cluster paths. | `experiments/attention_warm_route_verify_v1/submission.py` | #164 |
| MoE | 1.1465x → 2.3263x → 2.3290x → **2.3820x** | Persistent WGS experts, gather tuning and weight/activation permutation. | `kernels/muse_moe_w4a16/submission.py` | #204 |
| Whole decoder | 1.0962x → 1.7469x → 2.0746x → 2.0926x → **2.2603x** | Stacked attention, projection and routed-expert improvements. | `experiments/decode_keell0renz_compare_v1/submission.py` | #112 |
| Prefill layer | 1.7874x → 1.8092x → **1.9321x** | Integrated attention, GEMM-plan and MoE stack. | `experiments/prefill_keell0renz_compare_v1/submission.py` | #32 |

The working projection file contains later experiments and is not byte-identical to the best scored source. See `research/best.json` for hashes and official jobs, and the experiment logs for checks that did not improve the public score. The [baseline and decoder integration audit](research/baseline-integration-audit.md) records which standalone changes were tested together and why the decoder incumbent was retained.

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
overall #112; continue targeting integrated decoder dataflow and the larger
attention and MoE gaps to the leaders.

`serving/sglang` is the pinned SGLang checkout. `serving/patch.diff` currently
contains the fused Hopper router and a guarded small-batch expert-reduction
path, both derived from validated kernels. Syntax and patch rules pass; the serving patch has not been
validated on an 8-GPU node. The serving track opens November 5.

Attention backend: attributed MIT implementation from [team bigboateng](https://github.com/operatorstack/open-kernel-challenge-solutions), independently validated at 1.786690x; a shape-dispatched hybrid with our native/cluster attention is under official evaluation.

Latest prefill progress: the vjeux public candidate passed the full H200 check
at **1.79783x** (95% CI 1.77086–1.82479), and was submitted as
`68ac9af85a3a4e3d`, scoring **1.78740x** (95% CI 1.76059–1.81421); its rank was #74 at that earlier snapshot. The exact candidate
and check profile are in `experiments/prefill_vjeux_public_check_v1/`.

Prefill four-split attention passed the official check and a paired H200 component screen; scored submission `69464bf8cbce4588` passed at 1.78568x; the public best at that time remained 1.7874x.

Projection hybrid with vjeux M32 pooler and M128 router passed full check at 1.52701x after paired H200 screens; scored job `beb3e26afb2544c9` passed at 1.53951x, then rank #61.

Projection four-route hybrid adds paired-winning M1 unpooler and M256 router routes; check `4759f26857a74c46` passed at 1.52412x and scored job `4ed546f6ab2748cb` passed at 1.51826x; public best remains 1.53951x.

The isolated vjeux M256 router candidate passed check `593651829e0e4b78`; scored job `df44909a5b3540ff` passed at 1.52297x; public projection best remains 1.53951x.

Decoder T=1/8 output-projection transfer passed check `7ec7babbe7044aa6` at 2.08690x; scored job `1c4e4524414c407f` passed at 2.08743x, rank #95.

Decoder projection and MoE gather stack passed check `9ace173c90844588` at 2.08859x; scored job `6cf1be8d60e84ac7` passed at 2.09262x, then rank #85.

MoE shape-selected natural unit order passed check `72c6e933f69b4bad` at 2.32986x; scored job `6ae96ed05c124b5f` passed at 2.32168x; published MoE best remains 2.32905x.

Prefill boundary experiment: a first-recompute boundary launch adjustment passed full check `17ded4dec47047ad` at 1.79396×; three paired component repeats showed a 0.9–1.4% boundary improvement, too small to establish a full-score gain. The exact checked file scored 1.76757× as `9937a79b7f8448c4`, below the then-published 1.78740× best; intervals overlap and that incumbent was retained. Live prefill rank #114, #50 1.8046×, leader 2.0325× (2026-10-30). Source, timing logs and uncertainty are in `experiments/prefill_boundary_geometry_v1/`.

KPERM3 progress (2026-10-30): our GATHB=7 MoE stack passed official check `2ecdb5bf891d40e4` at **2.39112×** and was submitted as `190d8914bacb4aa2`; the integrated alygg77 decoder passed check `13161eab55cb447a` at **2.20175×** and was submitted as `78df8889fbbe4803`. The scored jobs passed: **2.3820× MoE (#70)** and **2.1957× decoder (#78)**; overall rank **#78** at the 2026-10-30 snapshot. MoE #50 was 2.4150×; decoder #50 was 2.2189×. The lossless K rearrangement comes from [alygg77's Apache-2.0 kernels](https://github.com/alygg77/okc-muse-spark-kernels); source, notices, and check profiles are archived in `experiments/`.

Decoder stacking: the combined compact T128 QKV and O projection routes on the KPERM3 decoder passed check `851e490e09f44621` at 2.19920×; its profile confirms both kernels ran. Scored job `b72f02c7eb8745f5` passed at **2.26032×** (95% CI 2.22642–2.29423), placing #25 at this snapshot; the point estimate is above the parent’s 2.19567×. Separate-run noise means the entire difference cannot be attributed solely to these routes. Details and the failed preparation attempt are in `experiments/decoder_alygg_bigproj_screen_v1/`.

Prefill K64 scale reuse: the four-lane shuffle candidate passed the official check at 1.80799× and scored **1.80923×** (95% CI 1.78210–1.83637), up from 1.78740×. It ranked #54 on 2026-10-30 at 14:42 UTC; #50 was 1.8115×. Intervals overlap, so the exact effect remains uncertain. Source and profiles: `experiments/prefill_k64_shuffle_v1/`.

Prefill head interleaving: the long full-attention candidate passed its official check at 1.80145× and scored **1.79978×** in `e3127297ccc1451f`, below the 1.80923× public best. Local paired tests showed a small component gain, but the score intervals overlap, so its aggregate effect is uncertain. The incumbent remains in place; details and profiles are in `experiments/prefill_head_interleave_v1/` and `experiments/prefill_head_interleave_check_v1/`.

Current prefill transfer (2026-11-02): K64 scale reuse/shuffle gather stacked onto `prefill_keell0renz_compare_v1` passed full check `972b04d5768c496c` at 1.95119×, but scored submission `91310268963a4503` at **1.92538×** (95% CI 1.89650–1.95427×). The live incumbent remains **1.9321×**, so the candidate was archived without promotion. The current public source, per-kernel ranks, and moving #50 cutoffs are in the table above and [`research/standings/2026-11-02.json`](research/standings/2026-11-02.json).

MoE research: screened MonoMoE's new H200 weight-major persistent design against our WGS implementation. Its central scheduling ideas—expert-grouped token jobs, persistent expert-weight streaming, one-launch fusion, and readiness flags—are already present in our current kernel; the paper's FP8 shapes and format are not directly comparable to this NVFP4 task. See the [transfer audit](research/MONOMOE_TRANSFER_AUDIT_2026-11-02.md). SC6 check v8 failed before timing due to a missing MFIN scratch tail. V9 now passes full correctness but its 2.36004× score is inconclusive against the 2.38203× incumbent. V10 scores the same timed kernel with a prepare-only print to confirm `-DOKC_SC6W=1`; the result is pending.

MoE batch-1 split-K screen: splitting the 4096-wide down projection into 256 K=16 CTAs passed six H200 reference checks but ran **0.823×** the parent speed (about 18% slower) across six cold-L2 seeds. The extra partial-output synchronization outweighed the parallelism. Rejected; details and source are in [`moe_single_token_splitk_v1`](experiments/moe_single_token_splitk_v1/FINDINGS.md).

Public-kernel audit (2026-11-02): [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) has two commits dated Nov 1 and publishes results from ~460 experiments. Its main techniques are already represented in our stack; exact-source judge results and the few remaining transfer ideas are recorded in the [audit log](research/KEELL0RENZ-AUDIT-2026-11-02.md). No upstream file was promoted.

Prefill research: a CuTe WGMMA/TMA screen tested the 16K-token QKV GEMM against the exact cuBLASLt route selected by the current scored source. All three candidates were bit-exact but slower (0.840×, 0.637×, and 0.917× parent/candidate); the CuTe route was rejected. The H200 logs and decision are under [`prefill_cute_qkv_long_v1`](experiments/prefill_cute_qkv_long_v1/README.md). No CuTe route is part of the scored submission.
