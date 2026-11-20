# Muse Spark Open kernels on a budget

My entry in the [Open Kernel Challenge](https://openkernelchallenge.dev/) kernel track (Oct 26 to Nov 5, 2026): CUDA kernels for Meta's Muse Spark Open decoder on one H200, scored as speedup over SGLang's own kernels.

All of it was run by Claude Code agents for about **$300–400 of compute in total**. As far as I can tell, the top entries spent around $4–5k, mostly on rented H200 time. I didn't have that budget, so this repo is also about getting as far as possible on a small one. Near the end, the limiting factor was compute, not ideas.

## Final results

Kernel track leaderboard, final.

| Board | What it is | Speedup vs SGLang | Rank | #1 |
|---|---|---|---|---|
| `muse_decode_layer` | whole decoder layer (decides the overall rank) | **2.275×** | 163 / 1296 | 2.614× |
| `muse_attention_decode` | gated attention over paged FP8 KV, decode + 8-token verify | **2.744×** | 78 / 1379 | 3.339× |
| `muse_moe_w4a16` | 256-expert MoE, NVFP4 weights × BF16 activations | **2.417×** | 154 / 1422 | 2.844× |
| `muse_bf16_proj` | skinny BF16 GEMMs + FP32 router, m = 1..256 | **1.592×** | 98 / 1740 | 1.728× |
| `muse_prefill_layer` | prefill layer (not ranked overall) | **1.932×** | 60 / 895 | 2.147× |

All final files are readable source; no encoded or compressed blobs. They are in [`kernels/<board>/submission.py`](kernels), with the judge job ids and hashes in [`research/best.json`](research/best.json).

## Where the money went, and how I kept it small

| Resource | Used for | Cost |
|---|---|---|
| Laptop RTX 3050 | compiling and correctness checks (it can't time Hopper wgmma/TMA code) | $0 |
| The challenge judge | ~20 unscored `okc check` runs per board per day on a real H200, plus scored submissions | $0 |
| [Modal](https://modal.com) H200s | paired A/B timing screens in the judge's Docker image | paid |
| Claude Code | the agents that wrote, screened and logged the kernels | paid |
| **Total** | | **~$300–400** |

What kept the cost down:

- **The judge was the main GPU.** Free daily checks on the real hardware were the experiment budget. Modal was only for screens that needed many paired runs.
- **Screen before checking.** A candidate went to the judge only after beating a same-code control by ≥1% in a paired Modal screen. Judge noise is ~0.6% per run, so anything smaller is just noise.
- **Agent discipline.** At most 2–3 subagents at once. Cheaper models for mechanical work (rebuilds, logging), the strong model for diagnosis. Short, narrow briefs. No polling, which re-reads the whole context on every check. Long jobs ran as scripts, not in live sessions.
- **One driver per question.** Each Modal run batched every variant and control into one container: one ~2.5-minute image prepare instead of one per variant.

## How it was built

An autonomous research loop in Claude Code ([`harness/`](harness)):

- [`harness/PROGRAM.md`](harness/PROGRAM.md) is the research program: a ratchet per board (keep a change only if it is correct on every shape and beats the best beyond the judge's 95% CI), every third experiment a "wild" design change, and a "when stuck" ladder (roofline, read public repos, switch boards, a fresh subagent, question an assumption).
- [`harness/claude/hooks/research_nudge.sh`](harness/claude/hooks/research_nudge.sh) is a stop hook. It blocks the session from ending and hands back a rotating strategy, so the loop ran overnight.
- [`tools/experiment.py`](tools/experiment.py) archives every candidate immutably and runs judge checks. [`tools/promote.py`](tools/promote.py) submits only a verified best.
- Every run is logged: [`docs/results.tsv`](docs/results.tsv), and over 770 experiment folders in [`experiments/`](experiments), each with its source, driver and `FINDINGS.md`.

## The kernels

- **MoE**: one persistent CUDA kernel per call. NVFP4 weights stream through a TMA + mbarrier ring and are decoded to BF16 in registers: a bit-permuted FP4 decode (7 ALU + 2 IMAD per 8 weights), PRMT scale decode and 6-bit packed scales. It uses stream-K work units, separate builds per batch-size range, and gate/up GEMM → SiLU → down GEMM → weighted top-8 sum fused in one launch.
- **Attention**: one fused kernel. Q/K norm, RoPE, FP8 cache write, split-KV FP8 attention (FP8 wgmma with an FP16-accumulated QK at q_len 8, a linear exp2 straight to E4M3), and the row sum taken from a ones-column in V. Partials merge through red.add atomics with a last-CTA finalize. Sliding verify at batch 2–24 uses cluster shared memory.
- **Projections**: register-direct swap-AB GEMVs on a lossless 12-bit BF16 encoding (an exponent window plus an exact escape list), with split-K across a cluster through DSMEM, and cuBLASLt plans picked by search for the large m.
- **Decode layer**: all of the above chained with Programmatic Dependent Launch. Residual/norm boundaries are fused into epilogues, pooler and router share one launch, and `prepare()` times each backend per token count on the machine it runs on.

## What didn't work

The negative results are probably the most useful part: [`docs/NOTES.md`](docs/NOTES.md) and the per-experiment `FINDINGS.md` files. Some highlights, all measured on an H200:

- A dynamic-claim small-batch MoE kernel ran at 0.55–0.67× of the incumbent; a `__threadfence` epilogue under saturated DRAM cost ~25%.
- An FA3-style look-ahead attention at 1 CTA/SM ran at 0.75×. Verifying the full context needs warp parallelism, not latency hiding.
- An 11-bit BF16 codec failed: the weights carry ~10.5 bits of entropy and the escapes ate the savings.
- Decoder projections were already near their floors; the slack was in fixed ~2.8 µs latency chains at kernel boundaries.
- A cross-SM handoff by polling a sentinel took 2.3–3.6 µs, slower than atomics + fence + a last-CTA ticket (~1.8 µs).

[`research/POSTMORTEM-PUBLIC-REPOS-2026-11-20.md`](research/POSTMORTEM-PUBLIC-REPOS-2026-11-20.md) lists what the top teams did that I didn't, from their post-competition repos.

## Layout

```
kernels/      final submission per board, with the task spec, reference and checks
experiments/  every candidate: source, Modal driver, logs, FINDINGS.md
dev/          lab scripts: Modal H200 runners, microbenchmarks, profiling
tools/        experiment archiving, judge checks, promotion, standings
research/     floors, rooflines, compliance audits, best.json, post-mortem
docs/         NOTES (facts and negative results), IDEAS, WILD, results.tsv, RULES, daily LOG
harness/      the Claude Code research program, stop hook, and an earlier multi-agent run
serving/      serving-track SGLang patches
```

## License

Apache-2.0 ([LICENSE](LICENSE)), as the challenge rules require for submissions.

## Credits

[vjeux/okc-2026](https://github.com/vjeux/okc-2026) · [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels) · [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) · [chenwainuo/okc](https://github.com/chenwainuo/okc) · [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) · [zhebrak/muse-spark-kernels](https://github.com/zhebrak/muse-spark-kernels) · Matsvei ([agents.kernarc.com](https://agents.kernarc.com))
