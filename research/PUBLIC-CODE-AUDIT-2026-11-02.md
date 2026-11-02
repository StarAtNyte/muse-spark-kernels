# Public kernel-code scan (2026-11-02)

Reviewed public challenge implementations and experiment notes while comparing them with our current five-kernel incumbents. Repositories were read at these exact revisions:

- [`vjeux/okc-2026`](https://github.com/vjeux/okc-2026), `f35a41ee1c13e316dabbf772830d47b23b453393` (Apache-2.0).
- [`operatorstack/open-kernel-challenge-solutions`](https://github.com/operatorstack/open-kernel-challenge-solutions), HEAD from 2026-10-26 (MIT).
- [`lee101/openkernels`](https://github.com/lee101/openkernels), `attention v84/v85` revision from 2026-10-28.

## What transfers and what does not

- **Projection / MoE / decoder / prefill:** the vjeux shape routes and the alygg77 KPERM3 layout are already in our scored stacks or in completed transfer experiments. The recent K64 prefill transfer onto the current full-layer source passed the full judge check, but scored **1.92538×** (95% CI **1.89650–1.95427×**) versus the incumbent's live **1.9321×**. It did not establish an improvement and remains archived without promotion; see [`prefill_keell_k64_transfer_v1`](../experiments/prefill_keell_k64_transfer_v1/README.md).
- **Attention:** lee101's `swk` design is the remaining distinct, shape-specific idea: for batch-16/32 sliding decode, independent warps fetch key chunks into registers and a small Hopper cluster performs the final merge, avoiding a serialized per-CTA chunk loop and global split-KV merge. Their notes report v84 gains of **2–4%** on those two shapes; v85's reordered and full-row variants lost and were parked. This is promising but narrow: it affects two of six attention loads, so even a 4% per-case win implies at most about **1.3%** geomean score gain if the other four loads do not move. Rebuild only the clean `swk` path on top of our current incumbent; do not copy the diagnostic file's host-side sweeps or stress harness into a submission.
- **Other public kernels:** operatorstack's published attention, MoE, projection and decoder runs are substantially behind our current judged kernels. They do not justify replacing or stacking any current route.

The v85 code itself is diagnostic: its `kernel_fn` invokes untimed sweeps and stress replays outside CUDA graph capture. That is useful development evidence but must be removed before any submission. Treat the reported v84 measurements as a hypothesis to reproduce against our incumbent, not as a score claim for our code.

## Current standing snapshot

The authoritative per-board snapshot is [`standings/2026-11-02-refresh.json`](standings/2026-11-02-refresh.json). We are inside prefill's top 50; the closest remaining cutoff is projections at **0.25%** faster, then decoder at **1.88%**, attention at **3.34%**, and MoE at **3.71%**. Overall rank is **#106**. The competition leaderboard moves continuously.
