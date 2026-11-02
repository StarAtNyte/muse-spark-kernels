# Chord W4A16 transfer probe: group-32 requantization rejected

Date: 2026-11-02. Reviewed Novita Labs' Chord W4A16 MoE kernels, current
checkout `6199d44` (Apache-2.0):

- https://github.com/novitalabs/chord
- https://github.com/novitalabs/chord/blob/main/docs/design.md
- https://github.com/novitalabs/chord/blob/main/docs/performance.md

Chord reports useful H200 grouped-prefill wins on its Kimi K2.5 workload,
including up to 1.26x for gate/up at 128 rows per expert. Its weights are
uniform signed INT4 with BF16 group-32 scales. Muse's weights are nonuniform
E2M1 NVFP4 with E4M3 block scales per 16 values. The high-level H200 schedule
is relevant, but its packed-weight representation is not a drop-in.

## Accuracy feasibility result

Before attempting to vendor or adapt its grouped kernel, dequantized the actual
challenge weights and fitted symmetric signed INT4 with a BF16 scale for each
32-value group (max-absolute/7, round and clamp to [-8, 7]). Ran the exact
challenge MoE reference with these quantized weights on 32, 128 and 256 tokens.

| Tokens | Output relative L2 | Challenge check |
|---:|---:|---|
| 32 | 0.165342 | fail; limit 0.015 |
| 128 | 0.164603 | fail; limit 0.015 |
| 256 | 0.164676 | fail; limit 0.015 |

W13 and W2 weight relative L2 error was 0.094973 and 0.094974. The simple
INT4 conversion is far outside the task's tolerance; do not integrate the
Chord packed format or claim its reported speedup transfers.

Probe ran on the public challenge image with generated task weights and
routing, Modal app `ap-WzNfRIOlmpxBcTmm7iLrLM`. The first probe attempt stopped
at setup because per-expert global scales must be passed as scalars; that
failed setup was corrected, and no timing was taken in either probe.

## What still transfers

Chord's most promising prefill idea is its semi-static token loop: always
execute tile 0, then guard only tiles `j >= 1`; its report compares this with
fully static and fully dynamic loops. Its WGMMA pipeline uses `wait_group 1`,
which is already present in our incumbent WGS consumer (`wg_wait<1>()` at the
per-chunk retirement points). A scheduling-only adaptation that keeps the
NVFP4 E2M1 values exact remains possible, but it needs a targeted H200 screen
before any kernel integration.

Do not directly transplant Chord's 1.26x figure: that comparison is for a
different model shape, routing distribution, quantization format and kernel
baseline. The useful transfer audit is the schedule, not the accuracy-breaking
weight conversion.
