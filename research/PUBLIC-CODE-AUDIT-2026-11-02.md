# Public kernel-code scan (2026-11-02)

Reviewed public challenge implementations and experiment notes while comparing them with our current five-kernel incumbents. Repositories were read at these exact revisions:

- [`vjeux/okc-2026`](https://github.com/vjeux/okc-2026), `f35a41ee1c13e316dabbf772830d47b23b453393` (Apache-2.0).
- [`operatorstack/open-kernel-challenge-solutions`](https://github.com/operatorstack/open-kernel-challenge-solutions), HEAD from 2026-10-26 (MIT).
- [`lee101/openkernels`](https://github.com/lee101/openkernels), `attention v84/v85` revision from 2026-10-28.

## What transfers and what does not

- **Projection / MoE / decoder / prefill:** the vjeux shape routes and the alygg77 KPERM3 layout are already in our scored stacks or in completed transfer experiments. The recent K64 prefill transfer onto the current full-layer source passed the full judge check, but scored **1.92538×** (95% CI **1.89650–1.95427×**) versus the incumbent's live **1.9321×**. It did not establish an improvement and remains archived without promotion; see [`prefill_keell_k64_transfer_v1`](../experiments/prefill_keell_k64_transfer_v1/README.md).
- **Attention:** lee101's `swk` design uses independent warps to fetch key chunks and a small Hopper cluster for the final merge. Upstream v84 notes reported 2–4% gains on batch-16/32 sliding decode, but our clean transfer did not reproduce them: v2 passed reference checks and measured candidate/incumbent speedups of **0.544×** and **0.584×**. Reject the transfer as integrated; no submission. This is a useful reminder that independent warps and cluster merging only help if their local register work and synchronization are substantially cheaper than the incumbent's pipelined split-KV loop.
- **keell0renz attention follow-up:** its public attention implementation differs from ours, but its q_len=8 full-attention route measured essentially at parity on our scored B16/B32 shapes. Removing only its dry cluster-finalizer prologue pass was then checked on six seeds and 24 randomized paired rounds for sliding q_len=1 B16/B32; all reference checks passed, but the timing signals were small and mixed. Keep the current route. Timings and exact candidate hash: [`attention_keell0renz_drycluster_remove_v1`](../experiments/attention_keell0renz_drycluster_remove_v1/README.md).
- **Other public kernels:** operatorstack's published attention, MoE, projection and decoder runs are substantially behind our current judged kernels. They do not justify replacing or stacking any current route.

The v85 code itself is diagnostic: its `kernel_fn` invokes untimed sweeps and stress replays outside CUDA graph capture. That is useful development evidence but must be removed before any submission. Treat the reported v84 measurements as a hypothesis to reproduce against our incumbent, not as a score claim for our code.

## Chord W4A16 MoE follow-up (2026-11-02)

Reviewed [`novitalabs/chord`](https://github.com/novitalabs/chord) at revision `6199d44a863a165139d2fd8387b307bbe21f1ca9` (Apache-2.0). Its H200-specific schedule chooses `mma.sync` swap-AB for small decode tiles, uses semi-static token tiles, and tunes tile depth/residency. The project's own H200 numbers are on Kimi K2.5 with INT4/group-32 scales, not Muse's NVFP4 E2M1 weights with group-16 E4M3 scales; its reported gains are therefore not a direct transfer estimate. See upstream [`design.md`](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/design.md) and [`performance.md`](https://github.com/novitalabs/chord/blob/6199d44a863a165139d2fd8387b307bbe21f1ca9/docs/performance.md).

The broad swap-AB/weight-major direction has already been tried here: `experiments/moe_wm_v22/analysis.md` records a correct **1.91656×** official run, well below the current **2.3820×** incumbent, with decode-32 especially weak at **1.342×**. A separate [`Humming transfer`](HUMMING-TRANSFER.md) also failed correctness and was much slower. So do not transplant Chord or Humming wholesale. I checked Chord's semi-static active-token tile idea against our current WGS source: `wgs_phase` computes `ntt = ceil(S.jC[j]/8)` and switches to `wgs_seg<PB, 1..8>`; that specialization sizes the accumulator and limits the output-tile loop to the active expert rows. This already implements the relevant compile-time tile-count dispatch, so the Chord rule is not a new experiment for our code. New MoE work should attack the measured WGS memory/issue gap or a different part of the critical path.

## Hopper TMA study and prefill screen result

The recent study [When and Why to Use TMA](https://arxiv.org/abs/2610.34907) characterizes H100 workloads and finds that TMA's issue overhead is not repaid on bandwidth-bound decode, while its CUTLASS GEMM and compute-bound prefill cases can benefit. Its absolute gains are not H200 predictions. The current prefill kernel uses tuned cuBLASLt for its large dense projections, so the bounded CuTe WGMMA/TMA comparison in [`prefill_cute_qkv_long_v1`](../experiments/prefill_cute_qkv_long_v1/README.md) tested the largest QKV shape directly against our exact selected parent route. All three tile/cluster variants were bit-exact but slower: 0.840×, 0.637×, and 0.917× (parent/candidate). Reject this generic CuTe example for this shape; the study's prefill result does not imply that replacing a tuned cuBLASLt path with any TMA kernel will win.

## Current standing snapshot

At 13:02 UTC the latest per-board snapshot refreshed ranks to projections **#66**, attention **#163**, MoE **#204**, decoder **#112**, and prefill **#32**; overall is **#112**. Scores did not change. The updated cutoffs are archived in [`standings/2026-11-02.json`](standings/2026-11-02.json). We remain inside prefill's top 50; the #50 gaps for projections, decoder, attention and MoE are +0.30%, +2.20%, +3.50% and +3.69%, respectively. The competition leaderboard moves continuously.

For the category leads, the score gaps from our current bests are still **4.7%** (projections), **26.8%** (attention), **16.1%** (MoE), **9.9%** (decoder), and **7.3%** (prefill). Treat top-50 entry and the #1 target as different optimization horizons; a threshold-sized change does not close the larger performance gap.
