# Long-prefill QKV descriptor randomized screen

The promoted prefill source and public judge image were used to revisit heuristic descriptors 24 and 184 for QKV at T=16384, N=2304, K=8192. Three inputs, 24 randomized-order graph replays per arm, 256 MiB read flush; every candidate output passed the incumbent-relative error bound. The duplicate incumbent graphs provide an A/A control. Candidate graphs call cuBLASLt directly while the incumbent uses `_mm`, so wrapper equivalence and independent reference correctness are not established.

| Seed | Incumbent A µs | A/A ratio | A / descriptor 24 | A / descriptor 184 |
|---:|---:|---:|---:|---:|
| 411 | 773.26 | 0.99959 | 0.99410 | 0.98418 |
| 719 | 778.75 | 1.00066 | 1.00724 | 0.99076 |
| 1013 | 777.79 | 0.99975 | 1.00738 | 0.99059 |

Descriptor 184 regressed in all three samples; descriptor 24 was mixed and its small lead in two samples is insufficient to justify a full-layer change. No official check or submission was used. Full output and exact descriptors: `modal.log`. The initial attempt (`first_attempt.log`) produced one row then an Xid 31/unspecified launch failure on the next seed; the cause was not established. Explicit graph and tensor release between seeds preceded the successful fresh-session rerun, which is consistent with but does not prove a graph-lifetime cause.

Reproduction: `modal run dev/modal/run_judge_image.py --file kernels/muse_prefill_layer/submission.py --drv experiments/prefill_qkv16384_random_v1/driver.py`.
