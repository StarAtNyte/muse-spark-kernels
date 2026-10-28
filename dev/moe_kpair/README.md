# Two-K64-chunk pipeline batches

Frozen parent experiments/moe_wm_v22/submission.py; diagnostic bundle, not a submission. Exact weight layout, BF16 activations, FP32 accumulation order, scale handling and output ownership retained.

Group two adjacent K64 chunks into one cp.async batch and one CTA synchronization. Four shared-memory chunk slots remain: one pair consumed while another pair is loaded. There are 32 steady-state barriers/commits instead of 64. Each pair's copies are waited on before all warps consume it; the next pair is written to the other slots, and the next barrier protects reuse. No change to weight bytes, number of CTAs, or shared-memory capacity. Tradeoff is fewer in-flight memory groups and altered instruction scheduling.

CPU ptxas: new gate/up 161 registers, down 159, no spills; baseline gate/up 157, down 160, no spills. Paired cold-L2 full-pipeline driver checks six fresh M128/M256 inputs for independent correctness and bit equality before timing. Original and candidate kernels share one module. Selector 128 chooses paired batching only.

Evidence: compile.log, screen.log, ../modal/drv_moe_kpair.py.

## Outcome

Completed app ap-5AgqSU9dS8hFQG4mFSirKo; six independent reference checks passed, all six outputs bit-identical. Ratios M128 0.9668/0.9980/0.9829, M256 0.9813/0.9971/0.9697. Reject: halving steady-state CTA barriers with this paired batching does not improve throughput. This does not isolate barrier instruction cost, because it also changes memory overlap; it rules out this specific capacity-preserving batching strategy. No official quota spent, all jobs terminal.
