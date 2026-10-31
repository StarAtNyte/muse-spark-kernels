# Forge attention reference profiling

Forge run `muse_attention_sliding_261031-170529-76ed` captured this Nsight Compute log from the scored attention reference (job `1793445787`). It covers the focused page-1 task's generated inputs; profiler clocks were unlocked and profiled kernel durations are not leaderboard timings. `log.txt` retains per-launch kernel names, launch dimensions, memory traffic, occupancy and stall metrics. The calibration and limitations of the challenge-aware correctness contract are recorded in `experiments/forge_attention_contract_v1/`.
