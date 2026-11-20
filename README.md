# Muse Spark Open kernels on a budget

My entry in the [Open Kernel Challenge](https://openkernelchallenge.dev/) kernel track (Oct 23 to Nov 5, 2026). The task was to write CUDA kernels for Meta's Muse Spark Open decoder that beat SGLang's own kernels on a single H200, inside the `lmsysorg/sglang:dev-muse-spark-open` image. The score is the speedup over SGLang: a geometric mean over each board's shapes, timed with CUDA graphs on fresh inputs and a cold L2.

The whole run cost about **$300–400 of compute**. Top submitters seem to have spent $4–5k. That gap set the pace. Near the end, the limit wasn't ideas but the GPU time to test them. For the same reason I didn't enter the serving track that ran after the kernel track: one scored serving run takes a whole 8×H200 node for up to 90 minutes. The patches in [`serving/`](serving) are early drafts that were never scored.

## Final results

| Board | What it computes | Speedup vs SGLang | Rank | #1 |
|---|---|---|---|---|
| `muse_decode_layer` | the whole decoder layer; decides the overall rank | **2.275×** | 163 / 1296 | 2.614× |
| `muse_attention_decode` | gated attention over a paged FP8 KV cache: decode and 8-token verify | **2.744×** | 78 / 1379 | 3.339× |
| `muse_moe_w4a16` | 256-expert MoE (32 local), NVFP4 weights × BF16 activations | **2.417×** | 154 / 1422 | 2.844× |
| `muse_bf16_proj` | skinny BF16 GEMMs (qkv_gate, o, pooler, unpooler) + FP32 router, m = 1..256 | **1.592×** | 98 / 1740 | 1.728× |
| `muse_prefill_layer` | prefill layer (not part of the overall rank) | **1.932×** | 60 / 895 | 2.147× |

The final files are readable source: no encoded or compressed blobs. They are in [`kernels/<board>/submission.py`](kernels), next to each board's task spec, reference and correctness checks. [`research/best.json`](research/best.json) records the judge job ids, SHA-256 hashes and 95% CIs.

## Per-shape results

All times are judge medians in µs. "Floor" is the judge's own speed-of-light estimate for the shape (compulsory bytes at peak HBM bandwidth). "Sliding" layers use a 2,048-token window; "full" layers attend over the whole context.

### Attention (job `526e460298bb4883`, 2.744×)

| Shape | Layer | Ours | SGLang | Speedup | Floor |
|---|---|---:|---:|---:|---:|
| decode, batch 1 | sliding | 4.8 | 20.8 | 4.30× | 0.1 |
| decode, batch 1 | full | 10.3 | 31.2 | 3.02× | 4.3 |
| decode, batch 16 | sliding | 7.4 | 26.8 | 3.61× | 1.6 |
| decode, batch 16 | full | 86.5 | 199.3 | 2.30× | 63.2 |
| decode, batch 32 | sliding | 9.5 | 31.3 | 3.29× | 3.3 |
| decode, batch 32 | full | 166.0 | 384.4 | 2.32× | 126.4 |
| verify, batch 1 | sliding | 6.4 | 23.6 | 3.66× | 0.1 |
| verify, batch 1 | full | 12.1 | 35.7 | 2.95× | 4.3 |
| verify, batch 16 | sliding | 9.7 | 29.4 | 3.02× | 1.9 |
| verify, batch 16 | full | 96.1 | 204.6 | 2.13× | 63.4 |
| verify, batch 32 | sliding | 11.7 | 34.2 | 2.92× | 3.7 |
| verify, batch 32 | full | 178.3 | 389.2 | 2.18× | 126.8 |

The sliding shapes are pure latency: a few µs of launch, merge and fence against a floor of almost nothing. The full-context shapes reach 62–76% of the bandwidth floor.

### MoE (job `5c2e8be4c4994a50`, 2.417×)

| Tokens | Ours | SGLang | Speedup | Floor | % of floor |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.6 | 77.1 | 3.13× | 11.4 | 46% |
| 8 | 62.2 | 142.1 | 2.28× | 37.3 | 60% |
| 16 | 90.7 | 188.7 | 2.08× | 52.5 | 58% |
| 32 | 125.0 | 211.0 | 1.69× | 70.8 | 57% |
| 128 | 172.6 | 459.0 | 2.66× | 101.3 | 59% |
| 256 | 207.1 | 617.6 | 2.98× | 108.9 | 53% |

The floor assumes 4.8 TB/s. In practice a cold-L2 contiguous stream reaches ~4.1 TB/s on this H200, and a 56 MB stream (batch 1) reaches only ~2.7 TB/s. Against that practical ceiling, the batch-16/32 kernels run at 86–87%.

### BF16 projections (job `1bdbbe7de3ea4351`, 1.582×)

Speedup by layer and row count m:

| Layer | m=1 | m=8 | m=16 | m=32 | m=128 | m=256 |
|---|---:|---:|---:|---:|---:|---:|
| qkv_gate | 1.58× | 1.53× | 1.44× | 1.39× | 1.42× | 1.14× |
| o | 1.35× | 1.27× | 1.22× | 1.17× | 1.11× | 1.07× |
| pooler | 1.55× | 1.54× | 1.51× | 1.44× | 1.44× | 1.20× |
| unpooler | 2.01× | 1.74× | 1.37× | 1.28× | 1.14× | 1.06× |
| router (FP32) | 2.81× | 6.87× | 3.75× | 3.00× | 2.95× | 2.58× |

At m ≤ 32 the qkv and pooler GEMVs run within 1–2 µs of the floor; there is little left to take. The o layer (n = 8192, k = 1024) is the weak spot at every m.

### Decode layer (job `348f0546c64b4e6f`, 2.264×; best scored 2.275×)

| Shape | Layer | Ours | SGLang | Speedup | Floor |
|---|---|---:|---:|---:|---:|
| decode, batch 1 | sliding | 68.3 | 174.1 | 2.55× | 40.0 |
| decode, batch 1 | full | 73.4 | 181.9 | 2.48× | 44.2 |
| decode, batch 16 | sliding | 129.4 | 268.2 | 2.07× | 76.7 |
| decode, batch 16 | full | 207.7 | 442.7 | 2.13× | 137.5 |
| decode, batch 32 | sliding | 140.6 | 255.0 | 1.81× | 84.5 |
| decode, batch 32 | full | 298.9 | 621.5 | 2.08× | 208.1 |
| verify, batch 1 | sliding | 103.8 | 260.3 | 2.51× | 62.3 |
| verify, batch 1 | full | 109.6 | 271.7 | 2.48× | 66.4 |
| verify, batch 16 | sliding | 233.3 | 525.1 | 2.25× | 126.6 |
| verify, batch 16 | full | 324.5 | 712.0 | 2.19× | 188.7 |
| verify, batch 32 | sliding | 307.7 | 736.6 | 2.39× | 143.8 |
| verify, batch 32 | full | 474.8 | 1092.8 | 2.30× | 267.2 |

In the batch-1 decode layer, MoE is ~30 µs of the ~68–78 µs. The four projection kernels take ~38 µs, and the rest is boundary latency.

### Prefill (job `f3b10ea957e04ea6`, 1.932×)

Between 1.89× and 1.96× on all eight shapes, from 518 µs (turn 1, sliding) to 12.6 ms (first turn, long context).

## Timeline

| Date | Decode layer | Attention | MoE | Projections | What happened |
|---|---:|---:|---:|---:|---|
| Oct 23 | 1.10× | 1.13× | 1.15× | 1.17× | first fused kernels on every board |
| Oct 27 | 1.73× | | | 1.54× | own projection stacks and per-T bucketed routes |
| Oct 28 | 2.08× | 2.58× | 2.33× | | rebased onto the strongest public Apache-2.0 kernels |
| Nov 2–4 | 2.26–2.27× | 2.60–2.65× | 2.38–2.42× | 1.59× | SC6 scales, KPERM3, WGS scheduling, FP32 storage, compact QKV/O routes |
| Nov 4–5 | 2.275× | 2.744× | 2.417× | 1.592× | FP16 QK accumulation in attention (allowed by the organizers' ruling), cleanup for readable final sources |

Over the twelve days that came to about 780 experiment folders, 115 scored submissions and about 300 unscored judge checks.

## Compute and budget

| Resource | Used for | Cost |
|---|---|---|
| Laptop RTX 3050 | compiling and correctness (it can't time Hopper wgmma/TMA code) | $0 |
| The challenge judge | ~20 unscored `okc check` runs per board per day on a real H200, plus scored submissions | $0 |
| [Modal](https://modal.com) H200s | paired A/B screens inside the judge's Docker image | paid |
| Coding agents (Claude Code, Codex) | | paid |
| **Total** | | **~$300–400** |

How the spend stayed small:

- **The judge was the main GPU.** The free daily checks were the experiment budget, and Modal was used only when a question needed many paired runs. One check gives per-shape medians, CIs and per-kernel profiles, which is more than most screens need.
- **Screen before checking.** A candidate went to the judge only after beating a same-code control by at least 1% in a paired Modal screen. Judge noise is about 0.6% per run, and nodes differ by a few percent, so smaller gains can't be told from noise.
- **One container per question.** The judge image takes ~2.5 minutes to prepare, so each Modal run batched every variant and its control into one container and one driver.
- **Agent discipline.** At most 2–3 subagents at once, cheaper models for mechanical work (rebuilds, logging), and short, narrow briefs. No status polling, since each poll re-reads the whole context. Long jobs ran as scripts, not as live sessions.
- **The Modal lab doesn't always predict the judge.** Wins on small shapes sometimes failed to transfer (different driver, different node), so every lab win still had to be confirmed with an `okc check`.

What more compute would have bought: more paired runs per idea (a 1% claim needs 6–8 judge runs per arm), the long Modal timelines needed for the structural rewrites below, and an entry in the serving track.

## Method

**Measurement.**
- Drivers mirror the judge: CUDA-graph replay, an L2 flush before every timed replay, the median of ≥40 alternating-order pairs, and the cost of an empty graph subtracted.
- Every screen includes a same-code control: the identical file with only the edit switched off. Code edits alone can shift timing by 1–4%.
- Correctness runs on every shape with the official `checks.py` tolerances, including skewed routing and nearby shapes that aren't in the test set.

**The research loop** ([`harness/`](harness)):
- [`harness/PROGRAM.md`](harness/PROGRAM.md) is the program. Each board has a ratchet: keep a change only if it is correct on every shape and beats the best beyond the judge's 95% CI, otherwise revert. Every gain on a component board is ported into the decode layer the same day.
- Every third experiment is a *wild* one: a design change with under 20% odds, such as a new fusion boundary, data layout or scheduler, written up in [`docs/WILD.md`](docs/WILD.md) first.
- After three rejections in a row comes the *when stuck* ladder: compute the roofline, read the public repos, switch boards, start a fresh subagent with only the logs, then question an assumption.
- [`harness/claude/hooks/research_nudge.sh`](harness/claude/hooks/research_nudge.sh) is a stop hook. It blocks the session from ending and hands back the next strategy, so the loop ran overnight.

**Bookkeeping.**
- [`tools/experiment.py`](tools/experiment.py) archives each candidate immutably (source, SHA-256, hypothesis) before running a judge check. [`tools/promote.py`](tools/promote.py) submits only a verified best.
- [`docs/results.tsv`](docs/results.tsv) has one line per judge run. [`docs/NOTES.md`](docs/NOTES.md) holds measured H200 facts and negative results, and [`docs/IDEAS.md`](docs/IDEAS.md) is the ranked backlog.
- Each folder in [`experiments/`](experiments) has the source, the Modal driver, the raw logs, judge profiles and a `FINDINGS.md`.

## The kernels

### MoE: [`kernels/muse_moe_w4a16`](kernels/muse_moe_w4a16)
- **Structure.** One persistent CUDA kernel per call, one CTA per SM. Every CTA builds the same per-expert token lists from `topk_ids`, so no separate routing launch is needed. It then runs gate/up GEMM → SiLU·up → down GEMM → per-expert RMSNorm → weighted top-8 sum in the same launch.
- **Weight path.** NVFP4 weights stream through a TMA + mbarrier ring and are decoded to BF16 in registers as the wgmma A operand. The FP4 decode is bit-permuted (7 ALU + 2 IMAD per 8 weights). E4M3 scales are decoded with one `PRMT` per pair, and scales are repacked to 6 bits (SC6) and reordered along K (KPERM3) so a single load serves a whole fragment.
- **Builds by batch size.** TINY (≤4 tokens), TINY2 (≤8) and MID (≤16) cover small batches. WGS, a stream-K scheduler over (expert, column-block) units with a fitted per-unit cost model, covers 17–256 tokens. The expert intermediates are stored in FP32.

### Attention: [`kernels/muse_attention_decode`](kernels/muse_attention_decode)
- **Fused front.** One fused kernel per call. Each CTA RMS-normalizes its Q rows, applies RoPE and rounds them to E4M3. The CTAs whose key range reaches the new tokens also build the new K/V and write them into the paged FP8 cache, so no CTA waits for another one's cache write.
- **Planner.** `_plan()` picks one of three regimes:
  - flat: all keys laid end to end and split evenly, with a producer warpgroup issuing copies;
  - per-request split-KV;
  - speculative builds for one request with page size 1.
- **Math at q_len 8.** QKᵀ is an FP8 wgmma with FP16 accumulators. P goes straight to E4M3 bytes through a linear exp2 in f16x2. The V tile carries 8 extra rows of E4M3 ones, so the softmax row sums come out of the PV wgmma itself. QK-norm bounds the logits, so a lazy per-row offset replaces the running max.
- **Merge.** Splits rescale to a fixed reference and add into a self-resetting workspace with `red.add` atomics. The CTA that completes a slice finalizes it: normalize, RMSNorm × sigmoid(gate), store. Sliding verify at batch 2–24 merges through cluster shared memory instead.

### Projections: [`kernels/muse_bf16_proj`](kernels/muse_bf16_proj)
- **Weight encoding.** BF16 weights use a lossless 12-bit encoding: the low byte plus a nibble holding the sign and a 3-bit offset into a per-matrix exponent window. Out-of-window values are fixed exactly from a per-warp escape list.
- **Kernel.** A register-direct swap-AB GEMV streams the encoded weights straight into `mma.sync` fragments. Split-K partials meet across a cluster through DSMEM.
- **Router.** FP32, computed as an exact BF16 hi/lo split.
- **Large m.** cuBLASLt plans picked by search.

### Decode layer: [`kernels/muse_decode_layer`](kernels/muse_decode_layer)
- **Composition.** The component kernels above are chained with Programmatic Dependent Launch: each kernel's weight loads overlap the tail of the one before it, and it waits with `griddepcontrol.wait` before reading that kernel's output.
- **Fusion.** Residual add, residual gate and the next RMSNorm are fused into projection epilogues or a single boundary kernel. Pooler and router share one launch at T ≤ 8.
- **Backend choice.** `prepare()` times each backend per token count on the machine it runs on: own kernel, cuBLAS or Triton for the projections, and own kernel or SGLang's Marlin for the MoE. It then pins the winners.

### Prefill: [`kernels/muse_prefill_layer`](kernels/muse_prefill_layer)
A grouped persistent NVFP4 MoE, own FP8 attention (with FA3 as a validated alternative), cuBLASLt-tuned dense GEMMs and fused residual boundaries. This board got the least of my time: the final file stays close to a public Apache-2.0 prefill file with a few of my patches.

## Where the remaining gap is

- **Attention.** The sliding shapes are 4–12 µs against near-zero floors, so what remains is launch, merge and fence latency. The leaders cut batch-1 sliding roughly in half with per-KV-head clusters and a DSMEM push merge, and used a byte-domain exp that writes E4M3 codes directly.
- **MoE.** It runs at 53–60% of the 4.8 TB/s floor. At batch 1, a phase-A spread across CTAs and ~2.3 µs of routing eat the budget. Barrier-free endings (per-token completion flags) are the next structural step.
- **Decode layer.** At batch 1 the layer is 68 µs against a 40 µs floor, and most of the excess is fixed latency chains of ~2.8 µs at each kernel boundary.

[`research/POSTMORTEM-PUBLIC-REPOS-2026-11-20.md`](research/POSTMORTEM-PUBLIC-REPOS-2026-11-20.md) lists what the top teams did that I didn't, read from the repos they published after the deadline.

## What didn't work

All measured on an H200; details in [`docs/NOTES.md`](docs/NOTES.md) and each experiment's `FINDINGS.md`.

- **Dynamic-claim small-batch MoE** (a new kernel with atomic work claims and a rotating epilogue warp): 0.55–0.67× of the incumbent. Static claims were just as slow. The `__threadfence` in the epilogue under saturated DRAM cost ~25%.
- **16 warps or a two-segment schedule for small-batch MoE:** 0.88–0.98×. Phase A already spreads at unit granularity.
- **FA3-style look-ahead attention at 1 CTA/SM:** 0.75×. Full-context verify needs warp parallelism, not latency hiding.
- **Exponential offload to the FMA pipe and FP16 probabilities:** 0.72–1.00×. The kernel was latency-bound in the QK → softmax → PV chain, not MUFU-bound.
- **An 11-bit BF16 codec:** the weights carry ~10.5 bits of entropy, and escapes ate the savings.
- **NVFP4 weight compression:** judge weights are uniform over the 16 codes, so there is nothing to compress.
- **A T = 8 fused pooler + router:** it saved a ~4.3 µs launch but lost 7–11% elsewhere.
- **KV prefetch before the PDL wait, L2 prefetch and deeper rings:** within ±1%.
- **Cross-SM handoff by polling a sentinel:** 2.3–3.6 µs to see another CTA's store, slower than atomics + fence + a last-CTA ticket (~1.8 µs).
- **A fresh TMA + wgmma kernel for the o projection:** it tied cuBLAS at m = 128/256 and was 7–10% slower below that.

## Running it

Each `kernels/<board>/` folder has `TASK.md` (the contract), `reference.py`, `checks.py` and `submission.py`. A submission defines `prepare(weights)` (untimed: builds the extension, repacks weights, tunes) and a CUDA-graph-capturable `kernel_fn`. It needs an H100/H200 (sm_90a), CUDA 12.x and the challenge's SGLang image. The Modal runners in [`dev/modal/`](dev/modal) run any submission plus a paired driver inside that image on an H200.

## Layout

```
kernels/      final submission per board, with task spec, reference and checks
experiments/  every candidate: source, Modal driver, logs, judge profiles, FINDINGS.md
dev/          lab: Modal H200 runners, microbenchmarks, profiling scripts
tools/        candidate archiving, judge checks, promotion, standings
research/     floors, rooflines, compliance audits, best.json, post-mortem
docs/         NOTES, IDEAS, WILD, results.tsv, RULES, daily LOG
harness/      research program, stop hook, an earlier multi-agent run
serving/      unscored serving-track drafts
```

## License

Apache-2.0 ([LICENSE](LICENSE)), as the challenge rules require for submissions.

## Credits

[vjeux/okc-2026](https://github.com/vjeux/okc-2026) · [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) · [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) · [chenwainuo/okc](https://github.com/chenwainuo/okc) · [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) · [zhebrak/muse-spark-kernels](https://github.com/zhebrak/muse-spark-kernels) · Matsvei ([agents.kernarc.com](https://agents.kernarc.com))
