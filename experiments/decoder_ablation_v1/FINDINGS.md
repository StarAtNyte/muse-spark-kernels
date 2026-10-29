# Whole-decoder removal ablation (2026-10-29)

Driver `dev/modal/drv_decoder_ablate.py` (CUDA 13 Modal H200, scored decoder source, cold L2, 24 alternating
replays). Each variant removes one component's launches; producers' outputs are replaced by cached tensors from a
real call. Event-timed medians (us), lab weights/router (not the judge's router seed):

| case | base | no MoE | no attention | no dense* | no glue* | only MoE* |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 95.2 | 61.4 | 91.4 | 55.7 | 54.7 | 24.7 |
| decode 16 sliding | 142.0 | 75.6 | 136.1 | 94.2 | 68.4 | 122.2 |
| decode 32 sliding | 183.2 | 81.6 | 174.8 | 130.8 | 150.3 | 84.4 |

\* Removing the routing glue or the pooler/router leaves the MoE with stale/garbage routing, so those columns also
change the MoE's work; only the no-MoE and no-attention columns are clean. The run then crashed on verify-1 for the
same reason (garbage expert ids), after these three rows.

Takeaways: the MoE is ~34/66/102 us of the layer at T = 1/16/32 (~35-55%), attention only 4-8 us. At T = 16/32 the
in-layer MoE runs at ~70% of its 4.0 TB/s floor; it is the decoder's main lever, and the same kernel drives the MoE
board's weakest loads (decode 16/32).
