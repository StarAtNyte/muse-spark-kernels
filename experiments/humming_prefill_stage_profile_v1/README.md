# Humming grouped-prefill stage timing

The prior `humming_prefill_grouped_probe_v1` measured two grouped expert GEMMs
plus separate SwiGLU at about 489 µs versus 231 µs for the incumbent MoE
component, despite excluding Humming's gather and normalization. This follow-up
repeats that comparison and separately captures the first GEMM, the torch
SwiGLU/multiply/conversion sequence, and the second GEMM on the same synthetic
route distribution and weights. It uses the same pinned Humming commit and
public H200 image; each stage has 16 cold-L2 graph replays.

Stage medians are diagnostic: individually flushed measurements and graph
overhead do not add exactly to full-path latency, and fusing operations can
change scheduling, numerical behavior and memory traffic. The SGLang stub,
precomputed routing/input normalization and comparison to the incumbent rather
than the full independent reference mean this is not an official score or a
submission candidate. The subprocess is limited to 300 seconds; raw output is
in `modal.log`.
