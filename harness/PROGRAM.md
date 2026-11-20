# Research program: Open Kernel Challenge (Muse Spark Open, H200)

## Repo mapping (read first)

The program below uses generic paths. In this repo they map to:

| Program name | This repo |
|---|---|
| `<problem>/submission.py` | `kernels/<problem>/submission.py` (read that folder's `AGENTS.md` and `TASK.md` first) |
| `<problem>/best.py` | frozen candidate in `experiments/<name>/submission.py`, named by `research/best.json` (experiment, job_id, sha256) |
| `results.tsv` | `results.tsv` (root; new runs). History: `research/submissions.json`, `experiments/*/status.json` |
| `IDEAS.md` | `IDEAS.md` (root). History: `research/experiment-priorities.md`, `research/NEXT-MOE-REWRITE.md` |
| `NOTES.md` | `NOTES.md` (root). History: `research/ENGINEERING.md`, `research/H200-LOWER-BOUNDS.md`, `research/FLOORS-2026-10-29.md` |
| `WILD.md` | `WILD.md` (root) |
| local GPU | RTX 3050 (correctness only). H200 timing: `dev/modal/run_cuda13_h200.py`, `dev/modal/run_judge_image.py` |
| `okc` | `~/.okc/bin/okc` (account StarAtNyte) |

Also follow `auto-research.md` and `HANDOFF.md`. Never stage unrelated work. No Claude attribution in commits.

---

## The challenge (read once, then act)

Write kernels for Meta's Muse Spark Open (256-expert MoE, NVFP4 expert weights, BF16 projections, FP8 KV cache) that beat SGLang's own kernels on ONE NVIDIA H200 SXM, inside the lmsysorg/sglang:dev-muse-spark-open image.

Score = speedup vs SGLang (geometric mean over each problem's shapes), timed with CUDA graphs on fresh inputs.

Problems (one self-contained submission.py each, CUDA inline is fine):

| problem | what | priority |
|---|---|---|
| muse_decode_layer | whole decoder layer (attention + router + NVFP4 experts + glue) | #1: decides overall rank |
| muse_moe_w4a16 | routed experts, NVFP4 weights x BF16 activations | #2: biggest chunk of decode_layer |
| muse_attention_decode | gated attention decode over paged FP8 KV, decode + 8-token verify | #3 |
| muse_bf16_proj | skinny BF16 GEMMs (qkv_gate, o, pooler, unpooler) + FP32 router, m = 1..256 | #4 |
| muse_prefill_layer | prefill layer | last: not part of overall rank |

Kernel track closes 2026-11-04 23:59 PT = 2026-11-05 13:44 Nepal time. Feature freeze 6 h before (2026-11-05 07:44 Nepal time). After the freeze: only resubmits of the best files.

## The judge (this replaces eval.py)

- `okc check <problem>/submission.py` is unscored, about 20 per problem per day. This is the experiment budget. Spend it.
- `okc submit <problem>/submission.py` is scored. The leaderboard keeps the best run, so resubmitting a good file is free upside.
- `okc leaderboard <problem>` shows the board and the current #1. Read it at the start of every session.
- Judge nodes vary: the same file can score a few % apart. A gain is real only if it beats the best by more than the 95% CI the judge reports, or reproduces on a second check.
- If a local GPU is available, use it for correctness and rough timing first, and send only candidates that pass locally to okc check. A non-Hopper GPU can't time wgmma/TMA code, so use it for correctness only.

## Files

- `<problem>/submission.py`: the file you edit for that problem
- `<problem>/best.py`: copy of the best judged file (never edit by hand)
- `results.tsv`: one line per judge run (format below)
- `IDEAS.md`: ranked backlog (expected gain, cost, status)
- `WILD.md`: unfiltered crazy ideas, before each WILD experiment
- `NOTES.md`: measured H200 facts and negative results (what didn't work and why)

results.tsv columns: time | problem | job_id | branch | idea | type(normal/WILD) | correct(x/y) | speedup | 95% CI | per-shape notes | verdict | why

## Rules

- Correctness on every shape first. A fast kernel that fails one shape scores nothing.
- No harness exploits: no caching results across calls, no detecting test inputs, no skipping work the reference does. They get rejected and waste the deadline.
- Kernels are Apache-2.0 by the challenge rules. Building on other teams' published Apache-2.0 kernels is allowed if you credit them in NOTES.md and the file header. Check the current rules page before you rely on this.
- NEVER STOP. Don't ask whether to continue, don't wait for a lead, don't end a turn on a summary. If you run out of ideas, follow "When stuck". The user may be asleep and expects results on return.

## The ratchet (main branch, per problem)

1. Read okc leaderboard, results.tsv and the current best.py for the problem you're working on.
2. Profile before you guess: per-shape times from the judge, local nsys/ncu if available. Find the slowest shape and the slowest sub-kernel. Work on that.
3. One hypothesis, one change. okc check it.
4. KEEP (commit + copy to best.py) only if it's correct on all shapes and beats the best beyond the CI. Otherwise revert. Then okc submit any new best.
5. Every improvement in moe / attention / proj gets ported into muse_decode_layer the same day. That board decides the rank.

## Wild branches (exploration)

Greedy tweaking of one kernel plateaus fast, so every 3rd experiment is WILD:

- Branch `wild/<name>`. Try something you'd give under 20% odds that changes the design (kernel structure, fusion boundary, data layout, scheduling), not a block-size tweak.
- Before each one, write 10 unfiltered ideas in WILD.md and pick the one least like anything in results.tsv.
- A wild branch gets up to 3 checks before it's judged.
- Seed list: see IDEAS.md, section "Wild".

## When stuck (3 rejected experiments in a row on a problem)

Do the first of these you haven't done recently:

1. Roofline it: compute the bytes moved and FLOPs per shape against H200 (4.8 TB/s HBM3e, 132 SMs). If you're already at more than 80% of the bandwidth roofline, stop tuning this kernel and move to fusion or a different problem.
2. Read the public repos of top teams (search GitHub for "Open Kernel Challenge" / "okc muse"). Many publish kernels, measured H200 facts and negative results. Add 5 ideas to IDEAS.md.
3. Switch problems. Time spent on a plateaued kernel is better spent on a weaker board.
4. Spawn a fresh subagent with only results.tsv, NOTES.md, IDEAS.md and the problem spec, not this conversation. Run its most unusual idea.
5. Question an assumption: launch count, CUDA-graph behavior, the fusion boundary, dequant placement, data layout, whether a library call (cuBLAS/FA3/Marlin) beats your kernel at some shapes.

## Priorities (in order)

1. A correct, submitted file on every board (a missing board costs more than a slow one).
2. muse_decode_layer: compose the best sub-kernels plus fused glue.
3. The weakest board relative to the leaderboard median.
4. Wild ideas.
5. Ratchet polishing.
