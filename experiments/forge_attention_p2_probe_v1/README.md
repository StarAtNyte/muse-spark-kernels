# Forge attention second targeted profiling probe

Reference job `1793447328` attempted `notes/p2_ncu.py`, first restricted to verify1 and then without scope restriction. Both attempts reported no kernels profiled and no report file, although Nsight listed `attn_kernel` as available. The job failed; its output is not a kernel timing or performance finding. See `log.txt` and the archived script. The earlier broad profile is in `experiments/forge_attention_ncu_recon_v1/`.
