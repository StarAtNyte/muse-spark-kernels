# claude_decoder_structural_v1: design only, no GPU run

## Outcome
No candidate was implemented and no Modal run was made. The parent is an 895,151-byte file whose MoE kernel has many interacting build variants. A substantive structural change cannot be written and checked there within one 300 s Modal function. The previous experiment (claude_decoder_next_v1) spent about 190 s of that budget on `prepare` alone. Nothing under dev/modal was copied, and no other repo file was touched. No GPU, submit, commit, push or subagent was used.

## Parent
`experiments/scored_decoder_fp32_ybuf_v4/submission.py`, SHA-256 `616187c9dac236851cc4b76bb65cd1cf9c85490db96f241a2fb1fa3d0fcff2cc` (readable source, FP32 `ybuf`, scored job 529df1f504434040 at 2.2386x, CI 2.205-2.272, 13/13 tests). `kernels/muse_decode_layer/submission.py` currently has SHA-256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`. That is the encoded-source file named in RULES.md, so it is not a baseline.

## Where the time goes (judge profiles of the parent, job 529df1f504434040)
| shape | total us | moe_fused | other large items |
|---|---|---|---|
| decode 32 sliding | 151 | 73.5 | 4 sk2 projections 57, attention 9.4, row boundaries 9.3 |
| decode 32 full | 318 | 61.3 | attention 175 |
| verify 16 full | 357 | 170.2 | attention 106, cuBLAS projections 62 |
| verify 32 sliding | 329 | 210.1 (T=256) | cuBLAS projections 86.5, attention 15.4 |
| verify 32 full | 478 | 184.4 | attention 187, cuBLAS projections 89 |

The reference's idealized decoder estimates (TASK.md) are 143.8 us for verify 32 sliding and 84.5 us for decode 32 sliding; they are model estimates, not proven floors for every implementation. The fused MoE profile is about 1.7–1.9x the standalone reference's modeled estimates (170 vs 101 us at bs128, 210 vs 109 us at bs256). These are not matched standalone measurements, so they do not establish HBM utilization, compulsory traffic, or the cause of the gap. The flat modeled estimate is consistent with a weight-traffic assumption but does not prove weight streaming dominates.

## Already rejected, so not proposed
These were found by searching NOTES.md and experiments/*/FINDINGS.md.
- MoE final-pass poll backoff (claude_decoder_next_v1): 1.002x.
- L2 prefetch during attention, PCX pre-wait depths, KV prefetch before `griddepcontrol.wait`, unpooler+finish merge via a global barrier, fused 16-row gle at T=8.
- Row-boundary kernel variants (decoder_rowpre_v1).
- Attention FA3-style look-ahead, polynomial exp and FP16 P offload.
- Routing T=128/256 projections through the `bfp` kernels: the parent's source comment `BFP_MAX_T = 32` records that this was measured 10 us slower than cuBLAS in the chain at T=256 and a wash at T=128.
- Dynamic-claim, ring and K-split MoE variants (SMOE v1-v3).

## Proposed structural change: stop re-streaming and re-dequantizing weight tiles per 64-token job (T >= 64 MoE "fat" path)
Hypothesis, not yet measured. This is not a knob sweep: it changes the job decomposition of the fat path.

Evidence in the source. `OKC_FT_T 64` and `OKC_BIGM 64` define a fat job as at most 64 tokens of one expert. `fat_stream` is organized as `job_of(u) -> (job j, unit-block ub)` with `UBS` unit-blocks per job. The interpretation that an expert above 64 tokens creates independent jobs needs verification against the actual job table and routing distribution. If those jobs reload and dequantize the same weight tiles, reusing tiles could reduce work; neither the duplicate traffic nor its cost has been measured.

Change. For experts with cnt > 64, do not split by token; split by N (weight column) tiles instead. A CTA that owns an (expert, N-tile) unit loops over all ceil(cnt/64) token groups against one dequantized/resident B tile:
1. Each (expert, N-tile) unit streams its NVFP4 weight tile from HBM exactly once.
2. The unit issues wgmma (m64nNk16) for g = 0..G-1 token groups with A taken from the staged BF16 x rows (swizzled K-major layout, as in the existing `is_nat` path). The accumulators live in registers one group at a time, and the epilogue per group writes the FP32 phase-A `inter` or phase-B `ybuf` rows exactly as today.
3. Phase-B completion counters (`doneA`, `ld_acquire` poll) keep their meaning: one counter per expert counts finished N-tile units instead of (job, ub) pairs. For hot experts the counts change from `UB_A` per job to `UB_A` per expert.
4. Cold experts (cnt <= 64) keep today's one-job schedule, so the T <= 32 builds and their tuned costs (`OKC_FCOST`, `OKC_FCOSTB`) are untouched.

Precision and rules. All accumulation stays FP32 (wgmma FP32 accumulators, FP32 `ybuf`). There is no BF16 partial storage and no K-split change. Routing is untouched, since the same `topk_ids` feed the same tokens. The source stays readable: it is a plain edit of the job decomposition in the embedded CUDA template, with no encoding.

Expected gain is unknown. The prior illustrative assumption that 50–70% of routes belong to experts with three or four 64-token groups is unmeasured; it must not be used as a forecast. Even with occupancy data, the saved bytes depend on the exact job-to-tile mapping, cache behavior and overlap, and route-weighted counts alone do not establish the fraction of weight work saved. The score impact must be measured on the full layer.

Risk. A hot expert's N-tile unit becomes G times longer, which raises tail imbalance across 132 SMs. A mitigation is to assign hot-expert N-tiles first (the LPT order already used by `S.forder`) and keep the unit count (experts x N-tiles) well above 132.

## Measurement to do first (about 20 s of GPU, no kernel edit)
1. In one `prepare`d process, replay the judge's routing generator for the T=128/256 verify shapes (any seed) and count (a) jobs per expert, (b) w and r above, and (c) the bytes re-read if every job streams its expert's weights. This is a CPU-only calculation on `topk_ids`. It answers whether the hypothesis can matter before any kernel is written.
2. If w * (1 - 1/r) < 0.2, drop the idea. Fall back to the next candidate below.
3. If it is >= 0.2, implement the change under `-DOKC_FATN=1` in a `_moe_build` variant for `p.M >= OKC_BIGM` only. Time it paired with the parent using the one-invocation harness from `claude_decoder_next_v1/driver.py` (swap `state['ma_ext']` entries, ABBA, 256 MB flush, two seeds per shape), and check with `reference.check_implementation` on the T=128/256 verify shapes, the 13 TASK.md tests and at least three seeds.

## Fallback candidate if the measurement says no
cuBLAS nvjet projections at T=128/256 (86.5 us at verify 32) against the standalone `bf16_proj` board times at m=256 (qkv 18.0, o 8.6, pooler 25.2, unpooler 22.7 us). The decoder's nvjet qkv_gate (31.3 us) is 13 us slower than the standalone custom kernel, but `BFP_MAX_T = 32` records that the bfp route was slower inside the chain at T=256. The reason is unmeasured: it may be the standalone kernel's activation layout or the cold-L2 state. Any attempt needs a profile of why, not another routing switch. A separate compliance check is needed that cuBLAS `torch.mm(out_dtype=float32)` does not split-K reduce in BF16. The output is FP32, so it is plausibly fine, but this was not audited here.

## Limitations
- Nothing here was executed, so every gain figure is a derivation or estimate, not a result.
- The hypothesis that the fat path re-streams weights per 64-token job rests on reading `fat_stream`/`job_of` and the `OKC_FT_T` definition, not on a trace of the actual job table at T=256.
- Tile and dimension details for the expert FFN (intermediate width) were not extracted, so the 15 MB per expert weight figure is inferred from the 109 us floor and may be off.

## Hashes
- Parent submission: `616187c9dac236851cc4b76bb65cd1cf9c85490db96f241a2fb1fa3d0fcff2cc`.
- Candidate source: none created.
- Current `kernels/muse_decode_layer/submission.py` (encoded, not used): `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`.
