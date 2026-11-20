# Muse Spark auto-research guide

Adapted to this Linux workspace from the MIT-licensed [Auto-research-agent](https://github.com/rohanarun/Auto-research-agent), inspected at commit `0d9f1674b9bfa506574cc5dda4698273f347d97f`. Its macOS application itself is not installed or running here. The user's instructions, current challenge rules and each task's `AGENTS.md` take precedence.

## Objective

Improve all five Muse Spark kernels towards the leaderboard leaders. Start with `research/H200-LOWER-BOUNDS.md`, `research/ENGINEERING.md`, the relevant task contract and current submission. Focus on a measured bottleneck or a new structural idea, including integrated decoder transfer; large changes are welcome when the evidence justifies them. Do not infer the official score from one component.

## Experiment loop

1. Check the live worktree, pending job handles and current relevant best. Preserve other work and existing logs. Inspect the actual scoring weights, case shapes, rules and current quota.
2. Record a concrete hypothesis, predicted affected cases and mathematical rationale, the exact source hash, and a falsifiable outcome. Estimate the case's current gap to both the ideal roofline and a relevant measured comparator. Specify cache, routing, representation and timing assumptions.
3. Freeze candidate and driver under a new `experiments/` directory. Make the smallest *informative* screen (this may be a radical design) on H200 with bounded runtime, fresh inputs and appropriate reference checks. Use `dev/modal/run_judge_image.py` for the public challenge image once its availability is verified; record any remaining image or harness mismatch. Keep raw timings and failed runs. Avoid launching duplicate live jobs.
4. Compare paired baseline/candidate on identical inputs and timing semantics, with fresh seeds and affected cases. Treat discrepancies, noise, clock differences and reference failures as evidence, not wins. Follow a component gain with end-to-end testing, especially for decoder and prefill.
5. Submit promising frozen candidates to the official check, preserve job IDs, complete profiles, correctness, confidence intervals, hashes and negative results. Compare the full weighted objective with the incumbent; uncertainty and different-run noise limit the conclusion. Submit improvements as authorized by the user. Preserve the checked source byte-for-byte.
6. Update the lower-bound record, engineering log and README score progression as evidence changes. Commit and push the improvement or research record without staging unrelated work.

Official checks and scored runs have separate quotas. The user has authorized official checks, submissions and Modal testing; use them judiciously. Never present a microbenchmark, ideal floor, static count or a starter prompt as an official score. Official judge output is authoritative for compatibility and scoring.
