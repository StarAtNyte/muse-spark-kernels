# H200 latency floors, bottlenecks and headroom (2026-10-30)

This is a living record. Append new measurements, source hashes, run IDs, seeds, and changes in assumptions; retain raw logs. The score tables below are the **official reference's speed-of-light estimates**, copied from each published run. They are idealized estimates, not proven attainable times or measurements of our hardware. They assume the reference's traffic and routing approximations, and omit launches, ramp, barriers, layout conversion and dependencies. Cache residency, alternate representations, repeated routing, and quantization semantics can change mandatory traffic; do not call a bytes/HBM estimate a universal mathematical lower bound without specifying those assumptions. Each table's `yours` is the published run, not a current rerun.

## Hardware and model

NVIDIA specifies H200 SXM at 4.8 TB/s HBM and 1,979 TFLOP/s BF16 **with sparsity**; use ~989.5 TFLOP/s dense as a theoretical peak, not the sparse value. [NVIDIA H200 datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23?xs=443995). Actual clock and achieved rates differ. For a specified compulsory HBM traffic Q and operations F, the optimistic roofline is `max(Q / B_peak, F / P_peak)`; only use it if Q and F are justified for the actual algorithm/cache state. Sequential dependent stages require a critical-path bound; one cannot blindly add overlapped profiler durations. A measured microbenchmark is an **empirical comparator**, not a proof that no better implementation exists.

Modal cold-L2 probes measured contiguous bulk at 3.8–3.9 TB/s including event overhead, 4.05–4.13 TB/s after an estimated launch correction at 340–425 MB; smaller 56 MB streams reached ~2.7 TB/s. Random 128-byte gathers reached 3.6–3.76 TB/s at 655 MB. Empty-graph/event costs in these particular probes are ~4.8–5.5 µs; they are harness-specific and cannot be added to official timings indiscriminately. Evidence: `experiments/h200_read_methods_v1/`, `experiments/h200_gather_bw_v1/`.

## Projection: QKV, output, pooler, unpooler, router

For M×K times K×N, reference uses `Q = NK*bw + 2MK + MN*bw`, `F = 2MNK`, with weight/output byte size bw=2 for BF16 and 4 for FP32 router; see `kernels/muse_bf16_proj/reference.py`. Router uses FP32 input conversion and a permitted hi/lo split; account for actual materialization separately. Peak-based arithmetic intensity is `F/Q`; the BF16 compute/bandwidth crossover is ~206 FLOP/byte using 989.5 TF/s and 4.8 TB/s. Small-M shapes tend toward streaming/latency; large-M can shift toward tensor throughput. This is a traffic model, not a strict floor for compressed weights or cached inputs. The wide2 sweep gives qkv M256 27.0 µs vs torch.mm 28.8 µs, o M256 16.2 vs current 14.0, o M128 13.9 vs 12.3; SGLang routes were stubbed, so no full projection comparison for those cases (`experiments/projection_wide2_sweep_v1/`). Priority: obtain matched-judge profiles by shape and target score-weighted gaps; don't extrapolate from stubs.

## Attention: QK, softmax, PV, merge

For batch B, effective attended context C, query length L, Hq query heads, Hkv KV heads, head width D, the reference counts `F = 4 B C L Hq D` for QK+PV and `Qkv = 2 B C Hkv D` bytes for FP8 K and V, plus query/new-KV/output traffic (see `kernels/muse_attention_decode/reference.py`). Its formula uses an intentionally shortened context share for uneven sequences; it is a conservative estimate, not the actual repetition's bytes. KV reuse across query heads is essential; page-table indirection, LSE/softmax, split merge and causal masking add costs. For small shapes, cold-cache gather comparators were: decode-1 sliding 8.99 vs 7.0 µs, full 16.62 vs 13.2 µs; verify-1 sliding 11.07 vs ~7.0 µs, full 19.38 vs 13.2 µs. These are not apples-to-apples proof of a floor (pure gather omits math, writes, indexing differences); ~1 µs for merge is a hypothesis, not measured necessity. Evidence: `experiments/attention_small_floor_v1/`. Decode-32 full is near the random gather throughput comparator; validate any design on official page-1 distributions.

## Routed MoE: routing, gate/up, activation, down, normalization

For each touched local expert H=E=4096, gate/up/down contain `3EH = 50,331,648` FP4 values. With one FP8 scale per 16 weights, the reference accounts `3EH*(0.5+1/16) = 28,311,552` bytes, ~28.31 MB per expert. Scale bytes are 1/9 of this total. With U distinct cold experts, ideal weight-only HBM time is `U*28.311552 MB / 4.8 TB/s` = `5.898 U µs`; route reuse means U, not number of tokens, controls unique weight bytes. This assumes each distinct complete expert is fetched once from HBM, no caching/compression or skipping, and ignores activations and all extra traffic. Total reference arithmetic is `2 R * 3EH` FLOPs for R local token-expert routes. Dependency from gate/up through activation to down can impose a stronger critical path than aggregate roofline; current kernel also pays routing, synchronization and output reduction.

| batch | distinct experts | expert MB | ideal weight-only µs at 4.8 TB/s | measured stream µs | measured current kernel µs |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 84.9 | 17.7 | 30.8 | 39.0 |
| 1 | 2 | 56.6 | 11.8 | 23.1 | 31.3 |
| 16 | 13 | 368.1 | 76.7 | 106.6 | 109.6 |
| 16 | 12 | 339.7 | 70.8 | 99.4 | 102.2 |
| 32 | 15 | 424.7 | 88.5 | 119.9 | 130.5 |

Evidence `experiments/moe_stream_floor_diag_v1/`; the stream reads the same weight and scale bytes but is a simpler, strided-access comparator, not a guaranteed attainable kernel. At bs32, NCU measured 3.27 TB/s, 57% scheduler issue-active and ~2.4 lane instructions per FP4 weight (`experiments/moe_wgs_stalls_v1/`). Those observations support testing decode/scale instruction reduction alongside improved streaming; neither implies memory nor instructions alone are the sole bottleneck. CTA timelines (`experiments/moe_timeline_v1/`) show phase-A tail, phase-B wait and final barrier. At bs1, scale ablation and overlap could matter; at bs16/32, aim to close the 3–9% gap to this particular stream, then challenge the stream itself using a faster access pattern.

## Whole decoder: critical path

The reference model sums compulsory estimates for dense weights, router, distinct local expert weights, KV and activations, and uses the maximum of aggregate bytes/peak bandwidth and aggregate FLOPs/peak compute (`kernels/muse_decode_layer/reference.py`). This is optimistic: dense → attention → router/experts → output imposes dependencies, while independent work can overlap. For each candidate, record the end-to-end latency as well as a component ablation; never sum overlapping PDL profiler rows. Clean no-MoE ablation on lab decode sliding gave base/no-MoE 95.2/61.4, 142.0/75.6 and 183.2/81.6 µs at 1/16/32 tokens; difference 33.8/66.4/101.6 µs, conditional on this particular ablation and routing. No-attention differences 3.8/5.9/8.4 µs. `experiments/decoder_ablation_v1/` notes invalid glue/only-MoE variants, so those cannot substantiate a bound. A standalone improvement transfers only if the integrated critical path benefits; test it end to end.

## Prefill: dense, attention, routed experts, boundaries

The reference uses the same aggregate formula as the decoder, with `T=B*L`, dense matrix sizes, touched experts and local routes, and attended KV tokens (`kernels/muse_prefill_layer/reference.py`). Each stage depends on prior outputs; its aggregate roofline omits these ordering costs and should not be presented as an achievable end-to-end time. The 16,384-token first-turn MoE NCU sample measured 2.89 ms, 92.6% SM throughput and 10.5% DRAM throughput; tensor saturation is plausible for that case, not proof all prefill MoE cases have zero headroom (`experiments/prefill_moe_ncu_v1/`). A long full-attention test measured ~4.95 TFLOP / 6.71 ms ≈ 738 TFLOP/s in its component harness (`experiments/prefill_expmode_v1/`); distinguish FP8 query/value arithmetic from the BF16 roofline. Official profiler rows overlap in some cases; measure the critical path, not their sum.

## Scoring sensitivity and priority

For load `l`, let `T_l = sum_i w_i t_i` and `S = exp((1/L) sum_l log(B_l/T_l))`, where `B_l` is the weighted baseline and L is the number of loads. If one case j drops by fraction r, its exact multiplicative score change is `(1 - r*w_j*t_j/T_l)^(-1/L)`. This is a hypothetical score sensitivity, not a prediction that such a speedup is physically available. `tools/prioritize.py --problem <name> --status <official status.json>` verifies the stored score reconstruction before computing it; use an exact public run instead of mixing case results from different jobs.

On our published MoE run, each of the six loads contains one case: reducing any one by 20% hypothetically raises the MoE score by 3.79%. On published prefill, a 20% reduction in first-turn sliding, turn-1 sliding, or turn-4 sliding hypothetically raises its score by 4.92%, 4.69%, or 4.54%, respectively. The 16,384-token first-turn sliding result has a 67.1% share of its load; however NCU evidence indicates that its MoE is already close to tensor saturation, so inspect the full critical path before assuming that reduction is feasible. Sources: `experiments/scored_moe_wgs_gathb7_v1/status.json`, `experiments/scored_prefill_vjeux_v1/status.json`, and `experiments/prefill_moe_ncu_v1/`. Prefill is a separate board.

## Official per-case tables

The following are verbatim numerical records from public scored runs, in microseconds; `sol` is the judge reference model, not an empirical minimum. Load aggregation uses challenge weights and its prescribed scoring; improving one low-weight case by 10% does not yield a 10% board gain. Check current evaluator and per-load weights before projecting score changes. Published runs have measurement uncertainty and may not represent the latest local code.

### Projection

Source: `experiments/scored_projection_vjeux_hybrid_v1/profiles/submission.beb3e26afb2544c9.txt`.

```text
scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
layer=qkv_gate load=decode 1 m=1                              14.16       9.19   1.540x       7.87  
layer=o load=decode 1 m=1                                      6.46       4.69   1.378x       3.50  
layer=pooler load=decode 1 m=1                                21.26      13.40   1.587x      13.99  
layer=unpooler load=decode 1 m=1                               6.94       3.75   1.850x       1.75  
layer=router load=decode 1 m=1                                10.40       3.69   2.817x       1.75  
layer=qkv_gate load=decode 16 m=16                            13.90       9.87   1.409x       7.93  
layer=o load=decode 16 m=16                                    6.59       5.13   1.286x       3.56  
layer=pooler load=decode 16 m=16                              21.09      13.44   1.569x      14.06  
layer=unpooler load=decode 16 m=16                            18.59      13.46   1.381x      14.06  
layer=router load=decode 16 m=16                              14.72       4.08   3.609x       1.81  
layer=qkv_gate load=decode 32 m=32                            14.35      11.64   1.233x       8.00  
layer=o load=decode 32 m=32                                    6.30       5.30   1.189x       3.62  
layer=pooler load=decode 32 m=32                              21.58      14.53   1.485x      14.14  
layer=unpooler load=decode 32 m=32                            18.43      15.29   1.205x      14.14  
layer=router load=decode 32 m=32                              15.25       5.10   2.990x       1.86  
layer=qkv_gate load=verify 1 m=8                              14.53       9.49   1.532x       7.90  
layer=o load=verify 1 m=8                                      6.56       5.04   1.303x       3.53  
layer=pooler load=verify 1 m=8                                21.31      13.59   1.568x      14.02  
layer=unpooler load=verify 1 m=8                               7.28       4.52   1.611x       1.76  
layer=router load=verify 1 m=8                                26.14       4.24   6.164x       1.78  
layer=qkv_gate load=verify 16 m=128                           17.02      14.33   1.188x       8.42  
layer=o load=verify 16 m=128                                   7.42       6.84   1.085x       3.99  
layer=pooler load=verify 16 m=128                             24.78      20.32   1.220x      14.64  
layer=unpooler load=verify 16 m=128                           19.58      18.32   1.069x      14.64  
layer=router load=verify 16 m=128                             16.10       5.67   2.840x       2.21  
layer=qkv_gate load=verify 32 m=256                           19.90      17.99   1.106x       9.77  
layer=o load=verify 32 m=256                                   9.28       8.64   1.074x       4.48  
layer=pooler load=verify 32 m=256                             30.59      25.20   1.214x      17.37  
layer=unpooler load=verify 32 m=256                           23.97      22.69   1.056x      17.37  
layer=router load=verify 32 m=256                             17.46       7.92   2.204x       2.68  
```

### Attention

Source: `experiments/scored_attention_plan16_v1/profiles/submission.b95cd80bd3e6497f.txt`.

```text
scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 1 page=1      20.96       5.02   4.176x       0.11  
batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 1 page=1      31.01      10.93   2.837x       4.27  
batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 16 page=1 spread=agentic      26.70       8.22   3.248x       1.65  
batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 16 page=1 spread=agentic     200.11      86.88   2.303x      63.18  
batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode 32 page=1 spread=agentic      31.57      10.02   3.151x       3.29  
batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode 32 page=1 spread=agentic     384.14     165.88   2.316x     126.35  
batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 1 page=1 q_len=8      22.94       6.52   3.521x       0.12  
batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 1 page=1 q_len=8      36.13      14.30   2.526x       4.28  
batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 16 page=1 q_len=8 spread=agentic      29.10      11.18   2.604x       1.86  
batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 16 page=1 q_len=8 spread=agentic     204.02      99.28   2.055x      63.39  
batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify 32 page=1 q_len=8 spread=agentic      34.58      14.79   2.338x       3.72  
batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify 32 page=1 q_len=8 spread=agentic     388.50     183.17   2.121x     126.78  
```

### MoE

Source: `kernels/muse_moe_w4a16/profiles/submission.88e09ad35c814c29.txt`.

```text
scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
bs=1 load=decode 1 sharding=ep8                               77.70      24.91   3.119x      11.42  
bs=16 load=decode 16 sharding=ep8                            191.82      97.71   1.963x      52.52  
bs=32 load=decode 32 sharding=ep8                            213.95     132.30   1.617x      70.76  
bs=8 load=verify 1 sharding=ep8                              142.88      63.66   2.244x      37.32  
bs=128 load=verify 16 sharding=ep8                           455.65     179.64   2.537x     101.34  
bs=256 load=verify 32 sharding=ep8                           622.98     220.01   2.832x     108.94  
```

### Decoder

Source: `experiments/scored_decoder_projection_gathb7_v1/profiles/submission.6cf1be8d60e84ac7.txt`.

```text
scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
batch=1 ctx=80000 dist=soft layer=sliding load=decode 1 page=1     170.75      76.40   2.235x      40.02  
batch=1 ctx=80000 dist=soft layer=full load=decode 1 page=1     182.03      83.35   2.184x      44.18  
batch=16 ctx=80000 dist=soft layer=sliding load=decode 16 page=1 spread=agentic     265.30     137.09   1.935x      76.72  
batch=16 ctx=80000 dist=soft layer=full load=decode 16 page=1 spread=agentic     430.80     206.50   2.086x     137.46  
batch=32 ctx=80000 dist=soft layer=sliding load=decode 32 page=1 spread=agentic     270.00     159.77   1.690x      84.51  
batch=32 ctx=80000 dist=soft layer=full load=decode 32 page=1 spread=agentic     615.55     304.75   2.020x     208.10  
batch=1 ctx=80000 dist=soft layer=sliding load=verify 1 page=1 q_len=8     268.46     115.28   2.329x      62.29  
batch=1 ctx=80000 dist=soft layer=full load=verify 1 page=1 q_len=8     278.70     122.39   2.277x      66.44  
batch=16 ctx=80000 dist=soft layer=sliding load=verify 16 page=1 q_len=8 spread=agentic     538.14     260.56   2.065x     126.61  
batch=16 ctx=80000 dist=soft layer=full load=verify 16 page=1 q_len=8 spread=agentic     729.02     349.51   2.086x     188.68  
batch=32 ctx=80000 dist=soft layer=sliding load=verify 32 page=1 q_len=8 spread=agentic     718.82     331.92   2.166x     143.81  
batch=32 ctx=80000 dist=soft layer=full load=verify 32 page=1 q_len=8 spread=agentic    1087.49     493.93   2.202x     267.18  
```

### Prefill

Source: `experiments/scored_prefill_vjeux_v1/profiles/submission.68ac9af85a3a4e3d.txt`.

```text
scored shapes (µs; speed of light is the reference's bound)
case                                                         sglang      yours  speedup        sol  
batch=1 ctx=80000 dist=soft layer=sliding load=turn 1 page=1 q_len=753    1082.05     572.75   1.889x     256.95  
batch=1 ctx=80000 dist=soft layer=full load=turn 1 page=1 q_len=753    1466.56     915.32   1.602x     496.69  
batch=4 ctx=80000 dist=soft layer=sliding load=turn 4 page=1 q_len=753    3334.75    1794.94   1.858x    1036.28  
batch=4 ctx=80000 dist=soft layer=full load=turn 4 page=1 q_len=753    4905.33    3120.94   1.572x    1992.42  
batch=1 ctx=0 dist=soft layer=full load=first turn page=1 q_len=16384   17597.62    9486.34   1.855x    5518.43  
batch=1 ctx=32768 dist=soft layer=sliding load=first turn page=1 q_len=16384   16930.46    8936.81   1.894x    5663.35  
batch=1 ctx=32768 dist=soft layer=full load=first turn page=1 q_len=16384   21577.82   12806.41   1.685x    7710.00  
batch=1 ctx=65536 dist=soft layer=full load=first turn page=1 q_len=16384   24516.83   15459.88   1.586x    9847.68  
```
