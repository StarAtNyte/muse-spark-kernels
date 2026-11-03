# Checked candidates awaiting scored submissions

> **2026-11-03 compliance hold:** The recommendations below predate the organizer clarification. Do not submit these historical decoder candidates: they contain encoded source and/or BF16 projection split reductions. The scored stack finished at 2.242× and also contains encoded source. Retain this queue for provenance only; use the [rule audit](compliance/2026-11-03-rule-audit.md) and complete source and precision audits on replacement candidates before any submission.

Audit time: 2026-11-03 03:37 UTC. The official CLI reported decoder scored quota **0/5 left**, with 6 private checks left. Quotas and ranks can change; query `okc problems` and `okc jobs` before submission. There were no live jobs at the audit. No candidate below has a scored submission recorded. Check scores are private measurements and their confidence intervals overlap the public decoder best **2.26032×**; none establishes a leaderboard improvement.

| Candidate | What changed | Full private check | Exact source SHA-256 | Next step |
|---|---|---|---|---|
| [`decoder_compact_tinywarp_stack_v1`](../experiments/decoder_compact_tinywarp_stack_v1/FINDINGS.md) | Compact T=128 QKV/O plus M<=8 tiny MoE warp order | `c411f676f4374f35`: 13/13, **2.26054×**, CI 2.22664–2.29445 | `febc0d6edc62ce08a698f8868747cff412bbc24a5c834dbf02c1ca88a581fbde` | First scored candidate when decoder quota returns; compare the exact file's hash before submitting. |
| [`decoder_stack_compact_qkvo_v1`](../experiments/decoder_stack_compact_qkvo_v1/FINDINGS.md) | Compact T=128 QKV and O only | `d3ec1a729a6c4107`: 13/13, **2.27051×**, CI 2.23645–2.30457 | `aa354bd2472393cfb407a50260dbc25543dab3763c4b45bcffca4f450ca98b6e` | Independent fallback scored candidate; paired selected-case gain 1.00294×. |
| [`decoder_tiny_original_warp_v1`](../experiments/decoder_tiny_original_warp_v1/FINDINGS.md) | M<=8 tiny MoE original warp order only | `42157e3923564abf`: 13/13, **2.25398×**, CI 2.22017–2.28779 | `aa21faa03140a1fabdf58381c2c0606e8024e17d54455fc603fc47e557196785` | Isolated fallback; the stack above contains it. Its [paired follow-up](../experiments/decoder_tiny_warp_fair_v1/FINDINGS.md) covered selected cases, not the whole score. |

The stacked candidate and components are alternatives, not three independent gains to add numerically. Verify quota, current judge status and exact SHA-256, submit one file, save the returned job ID immediately, and collect that same job before taking a further decision. Existing scored WST attention `e503062862744b16` passed at 2.63841×, below public 2.64939×; it has already been submitted and is not in this queue. The selected VP2 two-block attention source was publicly scored and promoted at 2.64939×.

Raw official `status.json`, `profiles/` logs and exact submission files are retained in the linked experiment directories. `research/best.json` records validation history; the board's public best and rank are recorded separately in the README and timestamped standings snapshots.


**Update:** Scored job `db40bea1008841dc` was submitted for the exact combined stack after quota reopened. Collect this job before deciding on the component fallback candidates.
