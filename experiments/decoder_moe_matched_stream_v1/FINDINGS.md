# Matched decoder MoE and stream measurement

CUDA 13 H200, incumbent `kernels/muse_decode_layer/submission.py`, app `ap-YhQkmNqKmFnpTu75dIhuDd`. The driver captures the actual decoder MoE launch and local expert IDs, then streams the same active experts' packed skinny (M=16) or SC6 WGS (M=32) copies. Timings alternate after a 256 MiB read flush, 20 graph replays per arm. All four complete decoder outputs passed the reference check.

| Batch | Seed | Active experts | Packed MB | MoE µs | Best stream µs | MoE / stream |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4001 | 9 | 254.8 | 79.44 | 66.56 | 1.194 |
| 16 | 4002 | 11 | 311.4 | 86.40 | 79.28 | 1.090 |
| 32 | 4001 | 12 | 330.3 | 100.62 | 84.94 | 1.185 |
| 32 | 4002 | 10 | 275.3 | 88.90 | 72.22 | 1.231 |

The stream probe reads each selected packed expert once, whereas the MoE includes computation, routing/synchronization, and potentially different access and cache behavior. It is a comparison point rather than a guaranteed lower bound. The measured gap is 7–17 µs for these samples, substantially narrower than an inference from the theoretical 4.8 TB/s bound. The decoder's preparation logged `BAKE rq skipped-budget`; the measured MoE kernel and selected routing were captured directly, but whole-layer timing and judge parity were not established. No candidate or official submission was made.

Reproduce with `modal run dev/modal/run_cuda13_h200.py --file kernels/muse_decode_layer/submission.py --drv experiments/decoder_moe_matched_stream_v1/driver.py`. Full logs: `modal.log`.
