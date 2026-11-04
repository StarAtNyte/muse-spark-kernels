# claude_moe_routing_reuse_v1: does the token-group weight-reuse hypothesis hold on real routing?

## Verdict
The hypothesis in `experiments/claude_decoder_structural_v1/FINDINGS.md` (w = 0.5-0.7 of routes on experts with more than 64 tokens, r = 3-4 jobs per hot expert, 35-50% of weight stream/dequant work removable) is not supported by the routing the decoder actually produces.

- Decoder T=128 (batch 16 x q_len 8): no expert ever exceeded 64 tokens in 32 routings (max count 64, which is exactly one job). The proposed change has nothing to act on.
- Decoder T=256 (batch 32 x q_len 8): 1-3 experts exceed 64 tokens in all 32 routings, carrying 18-63% of the local routes (median about 39%). Each hot expert has 2 jobs (never 3+), so r = 2, not 3-4. Only 4.2-11.8% of weight-stream jobs repeat an expert (median about 8%).
- Under the modeled one-stream-per-job accounting, the 1–3 duplicate expert streams represent 28–85 MB. Dividing those hypothetical bytes by peak 4.8 TB/s gives 5.9–17.7 µs of idealized bandwidth-equivalent time. This is neither a measured HBM saving nor an upper bound on latency savings: actual cache hits, achieved bandwidth, overlap, dequantization and scheduling all matter.

Nothing here was timed. These are routing and job-table counts plus a bytes calculation; no kernel was run or modified.

## What was measured, and how
One Modal invocation on one H200 in the judge image (`lmsysorg/sglang:dev-muse-spark-open`, torch 2.14.1+cu130, function timeout 240 s). The probe itself ran 4.5 s; wall time of `modal run` was 17 s. Raw stdout: `modal_raw.log` (128 `RUN` lines); parsed copy: `runs.json`; aggregation: `summary.txt` (from `analyze.py`).

Routing sources, both the challenge's own code, run unmodified:
1. Decoder board routing (what the parent sees): `kernels/muse_decode_layer/reference.py` `generate_input` (ctx 80000, q_len 8, spread agentic, dist soft, page 1), then `ref_front` (attention, residual boundary, h), then `route`, then top-8 of the biased score. `local_start` is the busiest 32-expert range, chosen as `case_fields` does. Cases: batch 16 and 32 (T = 128 and 256), sliding and full layer, input seeds 0-3, 4 routers each: 16 routings per (T, layer).
2. MoE board routing: `kernels/muse_moe_w4a16/reference.py` `generate_input(bs, 'ep8', seed, dist)` for bs 128 and 256, dist normal and skewed, seeds 0-15 (64 routings). The scored MoE shapes list only "bs 128/256, sharding ep8" (default dist, which is `normal`); `skewed` appears only as a bs 128 correctness test. The default-dist reading is an inference from the function signature.

Job accounting mirrors `route()` in `scored_decoder_fp32_ybuf_v4/submission.py` (read, not edited): an expert with cnt <= `SK_T`=8 gets one skinny job, otherwise `ceil(cnt / FT_T)` fat jobs with `FT_T`=64 (`OKC_FT_T`). Builds with `M >= OKC_BIGM`=64 take the fat path, so T=128 and 256 both do. `_MA_MMAX` is 384, so T=256 is a single launch with no token chunking.

### Does a job duplicate weights? Yes, in software
`fat_stream` maps each unit to (job j, row block ub) via `job_of`, and loads weights with `fat_issue(p, S.jE[j], ub, k, ...)`. The tile address depends only on the expert id `jE[j]`, the row block and the K stage. So two jobs of one expert issue independent TMA loads of identical bytes and run the same dequantization, differing only in which tokens they multiply. Phase A has `UB_A`=32 units per job and phase B `UB_B`=16 per job, so each job streams the whole expert (w13 and w2).

Correction to the earlier note: in this file `OKC_WGMIN` is 9, `is_nat()` needs `ceil(cnt/8) >= 9`, and a job has at most 8 token tiles, so `is_nat` is never true. The T=128/256 fat path in the parent runs `fat_compute` (mma.sync), not wgmma.

### Scheduling: where the duplicates land
The fat queue `forder` is sorted by `ceil(jC/8)` descending, stable by job index (LPT). Full 64-token chunks (key 8) lead; a hot expert's partial last chunk has a smaller key and runs much later. In the data, every duplicate job at T=256 is such a partial remainder (median 14-24 tokens): `summary.txt` shows 0 extra full-chunk jobs and 5-24 extra remainder jobs per group. So these repeats are not issued next to the first load, and a 50 MB L2 against 0.5-0.7 GB of streamed weights is unlikely to still hold the tile. This is a reading of the source, not a cache measurement.

## Data (median over runs, min-max in parentheses; per-run counts in `runs.json`)
Columns: local routes, distinct local experts, max tokens on one expert, fraction of routes on experts above 64 tokens, experts above 64, fat jobs, skinny jobs, repeat jobs, repeat share of all weight loads, and the earlier note's formula w*(1-1/r).

| source | routings | routes | distinct | max cnt | routes in >64 experts | experts >64 | fat jobs | skinny | repeat jobs | repeat share | w(1-1/r) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decoder T128 sliding, default router | 4 | 168 | 17 | 42 | 0 | 0 | 6 | 10 | 0 | 0 | 0 |
| decoder T128 sliding, alt routers | 12 | 197 | 18.5 | 42.5 | 0 | 0 | 7 | 12 | 0 | 0 | 0 |
| decoder T128 full, default | 4 | 161 | 18.5 | 43.5 | 0 | 0 | 6 | 12 | 0 | 0 | 0 |
| decoder T128 full, alt | 12 | 192 | 18.5 | 43.5 | 0 | 0 | 7.5 | 10.5 | 0 | 0 | 0 |
| decoder T256 sliding, default | 4 | 348 | 21.5 | 85.5 | 0.254 (0.218-0.501) | 1 (1-2) | 11.5 | 12 | 1 (1-2) | 0.046 (0.042-0.087) | 0.127 |
| decoder T256 sliding, alt | 12 | 410 | 23.5 | 92.5 | 0.393 (0.214-0.631) | 2 (1-3) | 12 | 12 | 2 (1-3) | 0.078 (0.048-0.118) | 0.196 |
| decoder T256 full, default | 4 | 344 | 20 | 83 | 0.468 (0.45-0.481) | 2 | 12 | 10.5 | 2 | 0.091 (0.083-0.091) | 0.234 |
| decoder T256 full, alt | 12 | 380 | 21 | 85.5 | 0.387 (0.183-0.538) | 2 (1-2) | 13 | 11.5 | 2 (1-2) | 0.084 (0.046-0.100) | 0.194 |
| moe_w4a16 bs128 normal | 16 | 172 | 22.5 | 40.5 | 0 (0-0.494) | 0 (0-1) | 5 | 17 | 0 (0-1) | 0 (0-0.053) | 0 |
| moe_w4a16 bs256 normal | 16 | 354 | 24 | 78.5 | 0.24 (0-0.716) | 1 (0-3) | 13 | 11.5 | 1 (0-5) | 0.039 (0-0.179) | 0.12 |
| moe_w4a16 bs128 skewed (not scored) | 16 | 318 | 4 | 126 | 0.885 | 3 | 6 | 0 | 3 | 0.40 | 0.443 |
| moe_w4a16 bs256 skewed (not scored) | 16 | 640 | 4 | 252 | 0.997 | 3 | 11 | 0 | 8 | 0.667 | 0.695 |

Only the `skewed` generator, which is not a scored shape, comes close to the earlier w and r guess.

The formula w*(1-1/r) is not the saved weight volume. Work is per job, not per route, so the repeated fraction is (repeat jobs)/(all jobs): 4-12% for decoder T=256, against 0.09-0.32 from the formula. By actual loads the idea falls below the earlier "drop under 0.2" threshold.

Sizes: expert FFN width 4096, w13 [8192, 4096] and w2 [4096, 4096] NVFP4 with 16-element block scales, so 28,311,552 bytes per expert (the earlier 15 MB estimate was about 2x low). Unique weights per routing: decoder T128 median 0.51 GB (18 experts, about 106 us at 4.8 TB/s), T256 median 0.59 GB (21 experts, about 124 us).

## Assumptions and limits
- Router weights are not the judge's. The judge picks `router_seed` in `weights_fields` from per-case `router_load` / `router_distinct` targets that are not in the local files. I used the default router from `generate_weights(5353)` (4 routings per case) and three arbitrary router seeds 1, 2, 3 ("alt"). My T=256 distinct experts (median 21) are above the roughly 18.5 implied by the MoE-board floor (108.9 us at 4.8 TB/s), so the judge's load may be lighter; fewer hot experts would make the idea look worse, not better.
- Only 4 input seeds, one context (80000) and the `soft` distribution; no outlier or mixed-spread cases.
- Routes use the reference top-8 of sigmoid score plus bias. The parent runs its own `_route` kernel, which may differ on near-ties (tolerance 0.05). Not compared.
- `ref_front` took 0.0-0.3 s per case on the H200. I did not independently verify that its attention path covers the full 80000-token context for these shapes (spread agentic may draw shorter lengths); I used the function exactly as the reference provides it.
- Not measured: L2 hit rate on repeated tiles, per-job time, dequantization ALU cost, tail imbalance, or any kernel timing. The 5.9-17.7 us figure is bytes over bandwidth; ALU redundancy is separate and unquantified. The 210 us kernel time is from the earlier note's profile, not from this run.

## Consequence
For these sampled T=128 routings there are no multi-job experts; sampled T=256 routings have a modest number. The proposed change has a smaller apparent opportunity than originally hypothesized, with added tail-imbalance risk, but its latency impact is unmeasured. The reference's idealized estimates are not proven floors and these observations do not identify the cause of the remaining timing gap.

## Files
- `probe.py`: routing generation, job accounting and bytes (also the Modal payload).
- `run_modal.py`: single-use runner (mounts only reference.py, checks.py, nvfp4.py of the two problems).
- `modal_raw.log`: raw stdout of the one Modal invocation.
- `runs.json`: the 128 parsed `RUN` records. `analyze.py`: aggregation (stdlib). `summary.txt`: its output.
- `claude.log`: empty file that already existed here, untouched. `__pycache__/`: bytecode from a local unit check of `probe.py` on a made-up count vector (not a result); a shell hook blocked its removal, so it can be deleted.

Reproduce from the repo root: `modal run experiments/claude_moe_routing_reuse_v1/run_modal.py --file experiments/claude_moe_routing_reuse_v1/probe.py`, then `python3 analyze.py modal_raw.log`.

No submit, commit, push, edit of other files or subagent was used.
