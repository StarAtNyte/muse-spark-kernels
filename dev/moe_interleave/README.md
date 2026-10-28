# Interleaved independent MMA output tiles

Frozen v22 large row-split path. Diagnostic only. Explicit counters measured ~19% wait and ~14–16% math-pipe throttle per active warp; source-level loop order is output tile, K16 substep, token-column fragment. Test K16 substep, output tile, token-column fragment to increase distance between dependent updates to each accumulator. Each accumulator still receives the same K products in the same order; BF16 decode, FP32 MMA, packing, routing, shared-memory capacity, CTA grid and asynchronous pipeline remain unchanged.

Tradeoffs: compiler may retain more raw weight and scale values, increasing registers; alternatively it may repeat shared loads. No assumption that source loop order determines final instruction order. CPU resource triage and full-pipeline timing decide whether the transformation helps. Driver six fresh M128/M256 cases checks independent reference correctness and bit equality, then 24 alternating cold-L2 graph replay pairs.

Evidence: compile.log, screen.log, ../modal/drv_moe_interleave.py.

## Outcome

Completed six reference checks, all bit-identical. New CPU ptxas gate/up152/down150 registers, no spills, versus baseline157/160. Full-pipeline ratios M1280.9952/1.0147/0.9962; M2561.0028/0.9966/0.9747. Reject: no consistent gain despite reduced registers and changed dependency ordering. No official quota spent; GPU run terminal. This result closes the tested loop-interchange variant, not every instruction-level schedule. Next work should move beyond local edits to this dataflow.
