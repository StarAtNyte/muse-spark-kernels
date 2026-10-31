# Short prefill MoE targeted stall counters

The preceding `prefill_short_moe_ncu_full_v1` run did not emit a stall
breakdown. This repeats the same reference-checked 753-token sliding case and
public H200 image while requesting the explicit per-issued-instruction stall
metrics used in our earlier H200 NCU study. The inner profiler is limited to
275 seconds; raw output and any missing-metric failure go to `modal.log`.
One-input counters with unfixed clocks are diagnostic, not a score result.

Run `ap-smircMhkIyPXzQwzXj4TWe` completed with NCU and driver return code 0;
the reference check passed. Among the requested metrics, wait and barrier
were largest at 1.17 and 1.14 stalled-warp cycles per issued instruction;
dispatch stall was 0.74, long scoreboard 0.49, GMMA 0.41 and short scoreboard
0.41. These are aggregate sampled categories on one input, not attribution to
individual barriers and not an additive latency breakdown. They motivate
examining synchronization and work balance, but do not justify removing any
barrier without a dependency proof and correctness validation.
