# ForgeKernel on Modal H200

The local ForgeKernel clone is at `/home/nitiz/Projects/Self/Challenges/ForgeKernel`, on branch `modal-h200`. It is based on OpenBMB/ForgeKernel commit `5030503d3ff71851aa34c14b9caf13ebfa9d9a78`. `modal-h200.patch` records the local integration and Muse prefill boundary task; apply it to that upstream commit with `git am` to reproduce the changes. The clone's `.venv` contains ForgeKernel, Modal, CuTe DSL and pytest; its local `socat` supports Claude Code's sandbox.

Modal delivery archives each candidate tree and runs one GPU job in a short-lived H200 container using `lmsysorg/sglang:dev-muse-spark-open`, with a default 300-second GPU job timeout and a serialized local job lock. The local and detached H200 smoke tests passed; 52 relevant configuration, delivery and timing-audit tests passed (one skipped). The public image and ForgeKernel's judge are not the official challenge environment; validate any resulting kernel with the official harness. ForgeKernel has a CLI, not a bundled web dashboard.

The campaign task is `tasks/muse_prefill_boundary`: it uses the promoted prefill boundary implementation as its reference, with 753-, 3012- and 128-row shapes, a 5% target, a 30-minute loop budget and one teacher round. The first snapshot failed calibration because its sealed reference tried to read a physical sibling file; the second uses the bundle's module import and passed calibration. The search is using paired cold-L2 CUDA graph timings. Nsight Compute initially could not lock GPU clocks; a later independent H200 probe succeeded with clock control disabled, and the integration now supplies that setting for newly generated jobs. Current run:

```sh
cd /home/nitiz/Projects/Self/Challenges/ForgeKernel
.venv/bin/forge status .artifacts/muse_prefill_boundary_261030-211439-b293
tail -f .artifacts/muse_prefill_boundary_261030-211439-b293/run.log
```

Per-job logs return on completion. A run's generated judge, timing protocol and any accepted candidate must be reviewed against the challenge's actual semantics and checked on the official judge before promotion. `local.env` is gitignored and contains machine-specific configuration; do not commit credentials.

A second task, `tasks/muse_attention_sliding`, wraps the exact current scored attention source (SHA-256 `554adb3886491fa686e950773b2d5c5cc759377eae0f3a7d3abf9f15fd353293`) and includes cache outputs in its interface. Its task definition passes `forge new --dry-run`; it has not yet been calibrated or executed. It covers only three page-1 sliding cases, so the official attention suite remains necessary for any candidate.
