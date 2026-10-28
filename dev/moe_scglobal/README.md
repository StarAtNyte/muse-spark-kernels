# Direct scale loads into registers

Frozen parent: `experiments/moe_wm_v22/submission.py`. Large-token row-split path only; diagnostic bundle is not a submission.

Each warp loads its exact uint2 scale words directly with `__ldg` at the start of a four-K64-chunk scale group, then reuses those words in registers. The new path eliminates the scale cp.async pipeline, the shared-memory scale reads, and the 12 KiB scale ring. Selected configuration dynamic shared memory decreases from 60 to 48 KiB per block. Weight and activation staging, FP4 decode, MMA order and output precision remain unchanged.

Tradeoff: synchronous global-load latency and extra registers can outweigh removed shared-memory traffic. Compare six M128/M256 inputs with retained baseline graphs, check exact output equality and independent correctness, then measure 24 alternating cold-L2 graph replay pairs. CPU-only ptxas resource triage runs separately.

Evidence: screen.log, compile.log, ../modal/drv_moe_scglobal.py.

## Outcome

Completed app ap-QiaQUE1kbOdYwNYt7xagpn. All six correctness checks passed with bit-identical output. Ratios M128: 1.0083, 0.9765, 0.9399; M256: 0.9404, 0.9456, 0.9198. Reject: direct synchronous scale loads regress, particularly M256. No official quota spent. CPU ptxas reports no spills but 162 registers for gate/up and 158 for down; the shared-memory reduction therefore does not unlock four CTAs per SM. Next discriminator: whether a 128-register budget can retain sufficient throughput to exploit the smaller shared-memory footprint.

## Four-CTA register budget follow-up

`cap128.cu` changes only the direct-scale variant's launch bound from three to four resident CTAs. CPU ptxas confirms 128 registers but introduces spills: down 56 bytes stores/96 bytes loads, gate/up 64 bytes stores/288 bytes loads (static reported instruction footprints, not runtime traffic). A paired H200 experiment measures whether the increased potential occupancy outweighs those spills and the synchronous scale load latency. Baseline remains unchanged. Bundle `probe_cap128.py`, driver `drv_moe_scglobal_cap128.py`, log `screen_cap128.log`.

Four-CTA follow-up completed: six checks passed, all bit-identical. Ratios M128 0.8829/0.8503/0.9172; M256 0.9295/0.9392/0.9106. Reject as well. Increased occupancy potential does not compensate for this variant's spills and direct-load schedule. Both GPU apps and the CPU compile are terminal; no official checks spent. A successful redesign must reduce live state without introducing these spills or preserve asynchronous scale overlap; the direct-load family is closed for now.
