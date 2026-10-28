# Attention full-verify H200 screen (2026-10-28)

Goal: explain the weak full-attention verify score before changing more code.

Reference is the public `attention_hybrid_verify_v2` source, Gluon `_g_full_verify`,
on one Modal H200. Compared candidate copies against the checked-in public source
with paired CUDA-graph timing and fresh cache seeds. All tested outputs were bitwise
identical (`relative_error: 0.0`); representative scaled/page-size correctness
checks passed.

| Variant | Batch 16 | Batch 32 | Decision |
|---|---:|---:|---|
| `BN=256`, 4 stages, 8 splits | ratios 1.000 / 1.005 | 1.000 / 1.000 | Reject: neutral |
| `BN=128`, 4 stages, 16 splits | 1.000 / 0.997 | 1.001 / 1.001 | Reject: neutral |
| 16-row parallel post-merge kernel | 1.006 / 0.999 | 1.004 / 1.001 | Reject: neutral |

Ratios are public-source latency divided by candidate latency, across seeds 539 and
1427. Batch 3 and 33 were also checked for collateral regressions; their ratios
stayed within about 1% of 1. These screens do not warrant a full official check.

Interpretation: tile loop count, split count, and the serial last-CTA merge are not
the dominant costs in this current architecture. The next attention hypothesis
must change the streamed K/V dataflow or work mapping, rather than retune these
parameters. Scored page size is 1 and physical cache rows are scattered, so a
contiguous TMA path is not directly applicable.

## Live score snapshot

`okc leaderboard --json` on 2026-10-28 reported: attention #260/880 at 1.8672x,
MoE #213/923 at 1.9698x, decode-layer #252/802 at 1.7156x, projection #98/1165
at 1.5177x. Top-50 cutoffs were 2.4858x, 2.3116x, 2.0860x, and 1.5331x respectively.

MoE's latest per-shape analysis shows decode-32 at 1.466x and its profiling report
attributes about 96% of that path to the two weight-major GEMM kernels. Routing and
finish together account for under 5%. Further work should target the W4A16 GEMM
dataflow, not the router or output epilogue.

## MoE tile follow-up

- H200 NCU was run with `--clock-control none` because Modal does not permit
  clock locking. On the scored batch-32 operator-stack `_om__w2` gate/up and
  down kernels, it reported 62.61% / 57.95% peak DRAM throughput, 75.11% /
  70.41% L2 throughput, 17.38% / 15.28% tensor-pipe active cycles, and 228 /
  114 MB DRAM reads. This supports investigating decode/instruction throughput
  and tile efficiency rather than assuming the device is at the HBM ceiling.
- Official private check `moe_bn128_v1` (job `f43c9c875e3847ad`) passed 5/5 but
  regressed aggregate from 1.972x to 1.861x; batch-32 decode changed from the
  previous 141.8 us to 149.0 us, and batch-128 verify regressed to 294.2 us.
  Reject BN128.
- `moe_bk512_v1` (job `e6b8cfd1e0224932`) failed Triton compilation on the
  batch-128 mixed-tile path. A second candidate isolated BK512 to batch 32 with
  a separate prepared layout (`moe_bk512_m32_v1`, job `aa601897e9db42f3`); it
  passed reference tests but failed compilation at batch-32 runtime. Reject the
  simple BK512 direction; no public scores changed.

Next MoE design should replace/decompose the low-utilization W4 decode inside
`_om__decode_tile` or change its dataflow without increasing shared tile size.
- M32-only shared activation staging + WGMMA (`moe_m32_wgmma_v1`, job
  `aa2b0a2e76f8429c`) passed 5/5, but measured 1.947x aggregate and 150.6 us
  on decode-32, versus the public hybrid's 1.972x and 141.8 us in its recorded
  run. Reject it; the current direct BF16 MMA path remains faster there.

The current day's new profiling/screens did not produce a faster public candidate;
no public submission was made. The M32 WGMMA alternative also fails to change the
working conclusion: the next redesign should focus on weight decode itself, not
just swapping the matrix instruction or enlarging tiles.

## MoE scale-decode representation screen

To test whether the Gluon NVFP4 weight path is spending significant issue slots on
per-tile scale conversion:

- `moe_prescale_v1` (`99f81363ae5a4cc6`): convert each FP8 group scale to BF16
  in `prepare`; keep E2M1 weights packed. Passed 5/5, aggregate 1.957x. It
  regressed the batch-32 decode from the public run's 141.8 us to 148.6 us, so
  the extra scale traffic erases the saved conversion work.
- `moe_rebias_v1` (`27f07d1e244945a4`): keep one-byte scales but rebase the
  exponent during prepare, then reconstruct BF16 scale bits with a general
  normal/subnormal-safe Gluon integer path. Passed 5/5 but scored 1.743x, with
  batch-32 decode 214.8 us. Reject the general bit-reconstruction path.
- `moe_rebias_fast_v1` (`98e1c0c73080409f`): guarded at prepare to require
  normal E4M3 scales and use a short shift/OR reconstruction. Passed 5/5 and
  scored 1.941x, but still regressed batch-32 decode to 146.2 us and batch-128
  verify to 234.9 us. Reject; hardware/compiler's existing FP8 conversion is
  already cheaper than these added integer operations in this mapping.

No public score changed. This makes scale re-encoding an exhausted branch. The
next candidate should parallelize K work or change the packed E2M1 decode, with
per-shape timing before integration.
