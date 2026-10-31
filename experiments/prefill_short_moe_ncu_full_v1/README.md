# Short prefill MoE full NCU profile

Follow-up to `../prefill_short_moe_ncu_v1/`: same scored prefill source and
753-token sliding case, but NCU `--set full` requests PC-sampling and source
metrics missing from the initial report. It validates against the task
reference before profiling and uses the public challenge image. Clock locking
is disabled; profiling replay and the stubbed/pinned local setup limit
interpretation. The inner process has a 275-second timeout, and the runner a
300-second timeout. Raw result or failure is saved in `modal.log`.
