# Humming W4A16 transfer probe — 2026-10-28

Humming's [Oct 19 announcement](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-10-21-humming.md)
describes SM90 indexed NVFP4 W4A16 support and workload-selected MMA/Stream-K
configurations. The large-batch results make it a plausible source of MoE
ideas, so I tested its native path against the challenge's actual generated
weights, group-16 E4M3 scales, per-expert global scales, and busiest-EP8
routing. Source revision: `vllm-project/humming` commit
`64ab616cc26e6d5e4000a8fc3022814d2abd90bb`.

The probe transformed the local 32 experts to Humming's layout, used indexed
Humming W4A16 for gate/up and down, then applied the challenge's SwiGLU,
post-expert norm, and route weights. It ran on an H200 for M=16 and M=256.
After correcting the harness to provide zeroed output buffers for Stream-K,
both outputs still failed the challenge reference badly (relative L2 1.444 and
1.569). Median CUDA-graph latency was about 515 us and 737 us, versus the
current integrated stack's official 106 us and 273 us on the corresponding
loads. The timing is diagnostic rather than an official paired score, but the
correctness failure alone rules out promotion.

Decision: do not vendor or integrate this implementation. The benchmark's
reported H200 MoE gains use a different workload/format mix and did not
transfer directly to this challenge. Return to the validated custom W4A16
route and its measured per-regime dispatch. The small H200 probe is in
`dev/modal/drv_humming_moe.py`.
