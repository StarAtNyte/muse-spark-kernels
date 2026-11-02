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

## Chord W4A16 MoE follow-up (2026-11-02)

Reviewed [`novitalabs/chord`](https://github.com/novitalabs/chord) at revision `6199d44a863a165139d2fd8387b307bbe21f1ca9` (Apache-2.0). Its H200-specific schedule chooses `mma.sync` swap-AB for small decode tiles, uses semi-static token tiles, and tunes tile depth/residency. The project's own H200 numbers are on Kimi K2.5 with INT4/group-32 scales, not Muse's NVFP4 E2M1 weights with group-16 E4M3 scales; its reported gains are therefore not a direct transfer estimate. See upstream [`design.md`](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/design.md) and [`performance.md`](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/performance.md).

The broad swap-AB/weight-major direction has already been tried here: `experiments/moe_wm_v22/analysis.md` records a correct **1.91656×** official run, well below the current **2.3820×** incumbent, with decode-32 especially weak at **1.342×**. A separate [`Humming transfer`](HUMMING-TRANSFER.md) also failed correctness and was much slower. So do not transplant Chord or Humming wholesale. The still-unmeasured kernel idea is the **semi-static tile-boundary rule**: keep tile 0 unconditional and guard only subsequent token tiles, driven by per-expert counts already available in the prepared routing metadata. It is worth a small paired H200 screen only if it can be added to our existing WGS path without replacing its weight decode or introducing a host sync; measure decode-16/32 first, where the MoE component dominates the whole decoder.

## Current standing snapshot

The latest per-board snapshot is [`standings/2026-11-02-live-refresh.json`](standings/2026-11-02-live-refresh.json), fetched at 11:10 UTC. Current ranks are projections **#65**, attention **#162**, MoE **#204**, decoder **#107**, and prefill **#31**; overall is **#107**. The public scores did not change between the earlier and latest snapshots. We are inside prefill's top 50; the closest remaining cutoff is projections at **0.25%** faster, then decoder at **1.88%**, attention at **3.34%**, and MoE at **3.71%**. The competition leaderboard moves continuously.

For the category leads, the score gaps from our current bests are still **4.7%** (projections), **26.8%** (attention), **16.1%** (MoE), **9.9%** (decoder), and **7.3%** (prefill). Treat top-50 entry and the #1 target as different optimization horizons; a threshold-sized change does not close the larger performance gap.
