# Forge attention reference baseline

The focused page-1 sliding attention run measured three reference samples: 0.0393, 0.0455 and 0.0439 ms as reported by Forge, a 15.8% min-to-max spread. Forge anchored to the fastest. `arch_baseline.json` records the selected result, including a separate 0.0106 ms device figure and profiling metadata; its `ms` field should not be confused with the device field or a challenge leaderboard time. Raw job logs for all three samples are retained here.

This spread makes small changes difficult to interpret. A promising candidate needs independent paired repeated timing and the official full attention checks. The generated input groups, focused task and challenge-aware checker do not cover all official workloads; see `experiments/forge_attention_contract_v1/`.
