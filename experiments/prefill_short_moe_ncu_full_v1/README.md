# Short prefill MoE full NCU profile

Follow-up to `../prefill_short_moe_ncu_v1/`: same scored prefill source and
753-token sliding case, but NCU `--set full` requests PC-sampling and source
metrics missing from the initial report. It validates against the task
reference before profiling and uses the public challenge image. Clock locking
is disabled; profiling replay and the stubbed/pinned local setup limit
interpretation. The inner process has a 275-second timeout, and the runner a
300-second timeout. Raw result or failure is saved in `modal.log`.

Run `ap-kdOJ5A4gpRfuVsnjN2XCtZ` is identified in the raw log; NCU and driver
returned 0 after 38 passes and the reference check passed. The occupancy report
shows one block per SM limited by both registers and shared memory, with 12
theoretical and 9.95 achieved active warps per SM. Compute throughput was
71.91% and DRAM throughput 40.46%. The report still did not include a sampled
stall breakdown; `--set full` alone did not resolve that missing evidence.
These are one-input replay counters with unfixed clocks, not a speedup proof.
