# Forge attention targeted profiling probe

Forge reference job `1793446922` ran `notes/p1_all.py` but failed to produce its Nsight report: the profiler reported no kernels profiled, with `attn_kernel` listed as available, then the script reported no report file. See `log.txt` for the exact command output. This is a failed profiling probe, not evidence of candidate correctness or performance. The earlier broad reference profile remains available in `experiments/forge_attention_ncu_recon_v1/`.
