# Short prefill MoE targeted stall counters

The preceding `prefill_short_moe_ncu_full_v1` run did not emit a stall
breakdown. This repeats the same reference-checked 753-token sliding case and
public H200 image while requesting the explicit per-issued-instruction stall
metrics used in our earlier H200 NCU study. The inner profiler is limited to
275 seconds; raw output and any missing-metric failure go to `modal.log`.
One-input counters with unfixed clocks are diagnostic, not a score result.
