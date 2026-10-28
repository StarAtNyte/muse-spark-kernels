# Applying KernelFoundry's search approach to Muse Spark

User direction: improve this challenge project; Modal H200 testing and profiling
are available. Explore PTX, CUTLASS, and CuTe DSL where the official worker
supports them. Use the existing official evaluator as the final authority.

## Verified environment and support

- Modal profile `baptinjean20` can run NVIDIA H200 jobs. The existing runner is
  `dev/modal/run_h200.py`; its CuTe image installs `nvidia-cutlass-dsl`.
- Inline PTX and PyTorch `load_inline` CUDA extensions already execute in
  officially passing submissions. Examples include `mma.sync`, `cp.async`,
  `prmt.b32`, and `griddepcontrol` in the archived MoE/projection sources.
- The official rules explicitly list CuTe DSL, CUTLASS, TileLang, FlashInfer,
  and DeepGEMM in `lmsysorg/sglang:dev-muse-spark-open`. This was confirmed
  from the rules on October 26, superseding the earlier uncertainty. Keep
  runtime activation work inside the timed call and compile before graph replay.
  Rules: https://openkernelchallenge.dev/rules
- KernelFoundry is an optimization framework, not the challenge submission
  service. Its evolutionary search and profiler feedback are transferable.
  Its Intel results and architecture profiles do not predict H200 speedups.

Upstream: https://github.com/isl-org/kernelfoundry

## Router probe, 2026-10-26

Candidate: `experiments/projection_router_tune_v1/submission.py`.
Official private job: `50ece49c24264d6e`.
All 9 correctness tests and 30 timed cases passed at **1.188518x**,
95% interval **[1.170690, 1.206346]**.

The prepare-time search compared twelve tile/split-K choices against the original
router, using synthetic inputs, graph replay, and a 3% selection threshold.
Every shape retained the original route. The aggregate difference from the
previous 1.182878x is within noise; **do not promote or submit as an improvement**.

Independent Modal paired sweep: `dev/router/modal_router_v2.log`.
Driver: `dev/modal/drv_router.py`. Each candidate passes the reference checker
before timing, with alternating replay order and a 256 MiB flush. All twelve
choices also passed batches 5, 9, 33, and 130, including outliers. No robust win.
The first Modal log is invalid for performance analysis: its driver checked
captured output before replaying the graph. The corrected driver explicitly
replays and synchronizes before checking output.

## Search priorities

1. Profile dominant dense GEMMs and routed-expert GEMMs on Modal. Record memory
   throughput, tensor-pipe utilization, register pressure, and launch/reduction
   costs before selecting a new implementation.
2. Explore substantially different data movement and reduction schemes rather
   than extending this unsuccessful router tile sweep. Inline PTX is already
   a viable official-worker backend. Evaluate Hopper WGMMA/TMA only where their
   setup and tile utilization fit the small-batch workload.
3. Test fresh seeds, outliers, nearby batches, and graph replay on Modal before
   spending official checks. Modal timings are diagnostic, not official scores.
4. Require a complete official passing integration and a meaningful gain before
   promotion. Decoder v7/v8 passed but did not beat decoder v3.

The live board snapshot is `research/live-standing-2026-10-26.json`; earlier
rank snapshots and README scores are historical.
