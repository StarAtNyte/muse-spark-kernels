# Prefill fused MoE at 16384 tokens: tensor pipe saturated (2026-10-29)

ncu on one first-turn layer call (ctx 32768, full; driver `dev/modal/drv_prefill_moe_ncu.py`, log `ncu.log`): moe_fused_persistent 2.89 ms, Compute (SM) throughput 92.6%, DRAM 10.5%, issue active 28%, 20.1 M gmma instructions, top stalls barrier 3.09 / wait 1.50 per issue (warps queuing behind the tensor pipe). The kernel is at the BF16 wgmma limit; only FP8 activations (not allowed outside attention queries) would go faster. No prefill MoE headroom.
