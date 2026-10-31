# Attention Nsight report import after wrapper fix

Forge reference job `1793447752` reran the p2 probe after the Modal Nsight wrapper stopped adding clock-control options to report imports. The report imported and the parser found 2,176 instructions; the SASS and orchestration self-checks reported `SASS-OK` and `ORCH-OK`. The orchestration summary reported E4M3-to-FP32 QGMMA, asynchronous copy instructions, and two named-barrier rendezvous patterns. These are profiling observations for one focused workload, not a speedup.

The separate structure self-check still reported `STRUCT-FAIL`: no per-PC execution counts were parsed, and it found no MMA/LDSM/STSM family in its segmented barrier regions. Do not treat its region summary as validated. The job process exited successfully despite this internal failed self-check. Full output and the probe script are retained here.
