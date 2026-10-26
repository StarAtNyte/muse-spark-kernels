# Muse Spark kernel challenge handoff

Updated 2026-10-24, approximately 11:50 UTC. Read this before continuing.

## Objective and authorization

User wants to win the Muse Spark Open Kernel Challenge, requested extreme kernel engineering, and explicitly authorized continued submissions: **“go on submitting.”** Continue testing and submitting validated improvements without repeatedly asking permission. Latest request was to write this handoff for other agents. The competition objective is unfinished; current results are improvements over stock, not evidence of a competitive winning position.

Do not spawn subagents unless a user or applicable AGENTS.md explicitly requests delegation. Do not rent paid GPUs or delete the user's large data folders without authorization. No cleanup is necessary for this handoff.

## Environment

- Workspace: `/home/nitiz/Projects/Self/Challenges/muse-spark`. Root is **not a Git repository**.
- Authenticated CLI: `~/.okc/bin/okc` v0.7.1, account **StarAtNyte**. Do not read or expose credentials.
- Local GPU: RTX 3050, 6 GB; not representative of the remote single-H200 judge. Roughly 22 GB free disk at last inspection.
- Default Python has no torch. `.venv` has Python 3.12 and Triton 3.6.0, supporting offline SM90 compilation without torch.
- Bundled disassembler: `.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump`.
- Read `kernels/<problem>/AGENTS.md` and `TASK.md` before editing that problem. All five problems are initialized.
- `serving/sglang` is a nested Git checkout pinned to `a55ab7d5ae1b`, branch `my-patch`.

## Verified results and submissions

Private speedups below are complete official checks with correctness passing. Public numbers are separate official scored runs; never present queued jobs as achieved public scores.

| Problem | Current canonical experiment | Private speedup | Latest scored job | Public status at handoff |
|---|---|---:|---|---|
| BF16 projections | `bf16_hybrid_v5` | 1.168898x | `3ee947021eae43b9` | Done, 1.174299x |
| Attention | `attention_fp8_hybrid_v7` | 1.316821x | `b9b6263e128341ad` | Just submitted; collect it |
| MoE | `moe_hybrid_v5` | 1.265151x | `33168954e1684bad` | Done, 1.268559x |
| Decoder layer | `decode_attention_hybrid_v4` | 1.153951x | `c3a588380db54444` | Just submitted; collect it |
| Prefill layer | Baseline diagnostic only | 1.000428x | None | No improved candidate |

**Last actions:** collected all previously pending jobs; promoted and publicly submitted attention v7 and decoder v4 using `tools/promote.py`. Both cleared the full correctness/count/hash gate. No new private checks were launched during handoff preparation.

The prior attention v5 public job `2c0559f533d145dc` completed at 1.238226x. The prior decoder v2 public job `4026b7baf4e04652` completed at 1.096193x.

## Authoritative artifacts

- `research/best.json`: canonical experiment, private job, hash, CI, and latest scored job. Public fields may lag a collection; check official archived status.
- `research/submissions.json`: scored provenance; old queued snapshots may need synchronization.
- `research/ENGINEERING.md`: detailed notebook and prior decisions. Some “pending” prose predates results in this handoff.
- `experiments/NAME/submission.py`: immutable archived source. `manifest.json` holds source hash, hypothesis, job ID; `status.json` holds full official records/profiles.
- `experiments/NAME/analysis.md`: per-load timings and profiler analysis where generated.
- `research/watch-events.jsonl`: bounded watcher journal.
- Canonical `kernels/<problem>/submission.py` files are copied by promotion, not manually chosen from a faster-looking partial case.

Never mutate an archived submission after its manifest is created. Create a new experiment directory for revisions. Do not blindly retry a POST after an ambiguous network error: inspect jobs/manifests first.

## Immediate next steps

1. Collect the two new public jobs. No watcher remains running: the prior 150-second watcher finished normally, with `watch_finished` in its journal. The handoff collector also finished. Old unified session IDs are no longer work to resume.
2. Analyze the newly completed experiments listed below, particularly attention v7, decoder v4, packed MoE v9, and baseline prefill. Update notebook/README with measured outcomes.
3. Keep MoE v5: v8's improvement is noise. Keep BF16 v5: both TMA variants lose overall.
4. Investigate merging attention v7 into the decoder. Decoder v4 uses the older attention v5 dispatcher; a new integrated candidate must preserve whole-layer rounding and pass its own entire official check.
5. Use prefill baseline profiles to choose a measured bottleneck before implementing a prefill optimization.

Useful commands (run from workspace):

```bash
python tools/experiment.py collect --pending
python tools/experiment.py collect EXPERIMENT_NAME
python tools/analyze.py experiments/EXPERIMENT_NAME
python tools/experiment.py check NEW_NAME path/to/submission.py --hypothesis 'one bounded change'
python tools/promote.py VALIDATED_NAME --submit
python -u tools/watch.py --seconds 150 --interval 20 --submit
```

Do not run concurrent collectors/watchers writing the same manifests. Watcher holds `.watch.lock`; promotions hold `.promotion.lock`. Avoid long blocking waits without commentary to the user.

`experiment.py` archives/hashes and submits privately, leaving five checks in reserve by default. It never automatically publishes. `promote.py` verifies official job identity, source SHA, passed correctness, no flags, full expected counts, and aggregate improvement; it prevents duplicate publication by parent experiment. Watcher additionally requires the candidate CI lower bound to exceed the current best CI upper bound. Do not bypass these gates to chase noise.

Expected test/timing counts: BF16 9/30; attention 16/12; MoE 5/6; decoder 13/12; prefill 5/8. A result wrapper saying `status: ok` is insufficient: require `passed: true`, zero return code, and all records passing.

## Latest completed private experiments

| Experiment | Private job | Correct | Speedup | Decision |
|---|---|---|---:|---|
| `attention_fp8_hybrid_v7` | `6e7717bfd5e04527` | Yes | 1.316821x | Promoted/submitted; CI [1.297069, 1.342776] |
| `decode_attention_hybrid_v4` | `0dd0d5d2f7aa4430` | Yes | 1.153951x | Promoted/submitted; CI [1.136641, 1.171260] |
| `decode_hybrid_v3` | `717244d81a0744f5` | Yes | 1.143994x | Superseded by v4 |
| `moe_m8_midbatch_v8` | `9d1bc7f8348d46d3` | Yes | 1.265244x | Essentially identical to v5; do not promote |
| `moe_packed_v9` | `a6e5098b3c3445b0` | Yes | 0.885958x | Reject globally; analyze profile |
| `bf16_tma_reuse_v9` | `dcae816b7c904d66` | Yes | 1.111973x | Reject globally |
| `bf16_tma_v8` | `b20bdf3f300043ef` | Yes | 1.102442x | Reject globally |
| `prefill_baseline_profile` | `a12664711e9a4297` | Yes | 1.000428x | Diagnostic baseline only |

## Engineering state and lessons

### BF16 projections

v5 uses FP32 vector GEMV for dense M=1, a fused hi/lo BF16 transposed split-K router, custom pooler for small batches, and cuBLAS for remaining regimes. Blanket custom dense paths have consistently regressed.

Rejected: transposed dense v3 0.845x, WGMMA v6 1.068x, skinny v7 1.139250x. TMA v8 used BM64/BN128/BK64, four warps/three stages, split-K. TMA reuse v9 used BM128/BN64/BK64, eight warps/four stages to reduce repeated weight reads. Both emit real TMA and WGMMA but lose to the v5 aggregate. v9 compiled at 58 registers, no stack, ~98 KB dynamic shared memory. Lower registers and newer instructions alone do not establish a win. Inspect paired timings, tile waves, repeated weight reads, and split reduction traffic before more tile sweeps.

### Attention

v5 fused QK RMS/RoPE/FP8 Q/cache stores; custom split-KV for sliding B<=4 or full QLEN=1; FA3 for other regimes. v7 uses native FP8 QK dot (PV BF16) and a selective dispatcher: sliding B<=4 or QLEN=1 with B*HKV*4>=192; full QLEN=1 or B<=4 with context bound<=131072; FA3 for losing large verify regimes. Read source for exact guards.

All-split-KV v4 passed but only 1.019x: small decode wins, large verification loses badly. FP8 QK v6 passed at 1.175689x but lost in large verify; v7 selectively retains the good regimes. Kernels use online softmax, FP32 accumulators, paged gathers, causal/sliding masks, and final BF16 rounding before RMS/gating. Offline compiler utility `tools/compile_attention.py` covers ten SM90 variants (page sizes, scales, sliding/full, decode/verify). FP8 masked loads require `other=0.0`; integer zero caused a real compilation failure.

Production SGLang attention already fuses parts of preprocessing that the kernel benchmark leaves unfused. Production SWA/cache metadata and slot-zero behavior differ. Do not blindly port this benchmark helper into serving.

### MoE

v5 retains Marlin, fuses post-expert normalization/reduction below 32 tokens (avoids a costly PyTorch fallback), and uses M16 expert alignment for >=32 benchmark tokens. v8 tested M8 at 32..64 and showed no meaningful aggregate gain.

Rejected native routes: WGMMA v4 correct at 0.580715x; singleton v7 correct at 1.079605x overall but its one-token native route was only 0.551x. Packed v9 is correct but 0.885958x overall. It loads original FP4 bytes as uint32 words, expands eight codes with inline PTX into BF16, uses original block scales and FP32 global scales, BM16/BN64/BK64, and retains v5 for other regimes. It reduced register use to 136/105 and zero stack but still lost. Do not assume decoding arithmetic/register improvements beat Marlin. Weight precision and stored format remain restricted.

### Decoder layer

v2 combined router (>8 tokens; keep FP32 router input <=8), fused QK/cache with **stock BF16 rounding before RoPE and again after**, and small MoE fused norm. v3 adds validated M16 alignment. v4 adds selective attention v5; now current canonical. Future attention v7 integration must retain those rounding boundaries and whole-layer correctness.

### Serving draft

`serving/patch.diff` was approximately 6,011 bytes over four files at last check; **not validated on 8 GPUs and not submitted**. Kernel track ends Nov 4; serving opens Nov 5 and ends Nov 19. Recheck official schedule before acting. Prize money is serving-track based; top 50 overall kernel competitors get serving priority.

Draft includes `fused_split_router.py`, guarded Hopper router dispatch for 8<M<=256 and weights (256,8192), guarded BF16 small-token fused post-expert normalization, and M16 Marlin alignment for 32..512 tokens with exact shape/device guards. It deliberately retains the existing production attention implementation. Do not infer an end-to-end serving gain from isolated kernel results.

Regenerate with `okc serve diff serving`; it includes new files through intent-to-add. Verify `git -C serving/sglang diff --check`, Python syntax, and serving patch rules. Allowed changes stay within `python/sglang`, excluding prohibited model-loader changes; patch limit 720 KiB, no renames/mode changes, no added line over 400 chars. Read actual rules before submitting.

## Measurement and compliance boundaries

Official judge: one H200, paired stock/candidate CUDA graphs, cold L2, randomized unseen correctness inputs. Compare each candidate against stock within its own run; absolute latencies across runs are not reliable selection criteria. Score is the prescribed aggregate of weighted load sums, not the best individual shape.

Keep weights in native precision/packed FP4. Lossless repacks are allowed; do not persist pre-dequantized FP4 weights as a shortcut. Router FP32 hi/lo BF16 decomposition is allowed. Do not prune keys/experts, inspect harness internals, cache activation results, or special-case deterministic inputs. Weight-only preparation can be untimed; activation work stays timed. Consult TASK/AGENTS for complete rules.

Offline compilation proves code generation, not accuracy or speed. `cuobjdump` static SHARED is not total dynamic shared allocation; inspect Triton metadata too. `tools/bench_projection.py` is an unexecuted-on-H200 development benchmark, not an official score. No paid H200 host is currently configured; official checks are available, so this is not a blocker.

The user also supplied https://arxiv.org/abs/2607.17979 as motivation. Do not invent findings from that paper; read it through browsing if using its claims. Existing engineering notebook links Hopper tuning and Triton persistent matmul references; Blackwell CLC/TMEM/native-FP4 assumptions do not apply to H200.


## UPDATE 2026-10-24 (second session) — custom CUDA kernels, read this first

Strategy change: stock-kernel wrappers (Marlin/FA3/cuBLAS + small fixes) cap at ~1.2-1.3x. Leaders are ~2x+ on MoE/attention/decoder
(rank-50 targets: bf16 1.33, attention 1.97, moe 1.99, decoder 1.72). The new approach is **hand-written CUDA streaming kernels**
(cp.async rings, mma.sync, weights repacked in `prepare()` into fragment order) plus an **in-submission H200 "lab"**: `prepare()` (or the
first eager `kernel_fn` call for attention) runs config sweeps/ablations with CUDA-graph timing and `print`s results; the log comes back in
`experiments/<name>/profiles/submission.<job>.log` (look for `LAB`/`LABD`/`LABA` lines). One check = one experiment batch.

Local toolchain (no nvcc/torch by default): `.venv-torch` (torch cu128 + triton), `.cuda_home/home` (nvcc redist + headers). gcc 15 breaks nvcc, so
device code is compiled with `clang++ -x cuda --cuda-gpu-arch=sm_86` and called through ctypes (see `dev/*/test_*.py`) on the RTX 3050.
sm_86 has no FP8 mma/cvt: write f16/bf16 paths locally; H200-only instructions are guarded by `__CUDA_ARCH__`.

Work dirs (all generate single-file submissions, immutable copies go to `experiments/` via `tools/experiment.py check`):
- `dev/moe_wm/` — weight-major NVFP4 MoE (swap-AB mma, denormal-linear FP4 decode, per-warp cp.async ring, runtime token-tile branch, multi-pass for >32 tokens/expert, PDL).
  `wm_cuda.py` (CUDA + repack), `wm_glue.py`, `wm_lab.py`, `build.py` (MoE problem; env `WM_CFG_SPEC`, `WM_LAB`), `build_decoder.py` (decoder = v4 + wm experts), `test_wm.py` (local correctness).
  Results (private checks): v5 Marlin 1.265x -> v10b 1.472x -> v12 1.580x -> **v13A 1.680x** (public job a925c611492f4514). Config tuple = (tiles/warp, warps/CTA, ring).
- `dev/dense/` — bf16 dense GEMM for qkv_gate/o/pooler/unpooler/router (`dense_cuda.py`, `build_bf16.py`, `lab_bf16.py`); `bf16_a.py` / `bf16_lab.py` submitted.
- `dev/attn/` — CUDA split-KV flash-decode (`attn_cuda.py`, f16 mma path, locally validated on 7 reference cases; H200 uses hw fp8->f16 cvt via -DFD_HW_CVT);
  `build_attn.py` (env `FD_POLICY`, `FD_LAB`), `attn_a.py`/`attn_lab1.py` submitted. Reuses v7 Triton prepare + merge kernels.
Pending check jobs at handoff: moe lab1/lab2/lab3 + v14pdl, decoder_wm_v5, bf16_dense_a/lab1, attn_fd_a/lab1 (see `okc jobs`).
Check budgets/day: moe 20, bf16 20, attention 10, decoder 10 (queue wait is ~10 min per job).

## UPDATE 2026-10-24 late (third pass)
Private bests: MoE 1.80x (moe_wm_v15B public ok; v17 = race-hardened equivalent, use as base), decoder 1.382x (decoder_all_v1: v17 MoE + fd2 attention; public cd8449...), attention 1.394x (attn_fd2_c; public ce9f79...), BF16 1.17x (dense kernels only match cuBLAS; dropped).
Lessons: (1) cross-kernel data under PDL must be read with __ldcg and after griddepcontrol.wait at kernel top (v16 failed rechecks otherwise). (2) mma.sync e4m3 is emulated by ptxas on sm_90 (no speedup). (3) cp.async activation rings (ring=2, dense x-ring) are slower for M>=8. (4) NT=8 tiles/warp, CUDA finish kernel, L2 bulk prefetch, fixup-reduce: no gain. (5) fused sliding kernel (slide1) correct but not faster than fd2.
Open ideas: persistent MoE with cross-unit prefetch (~+8%), verify-attention (L=8) rewrite (currently FA3/v7, 1.0-1.4x), L2 prefetch of next GEMM weights in decoder.
Builders: dev/build_decoder_all.py (env WM_CFG_SPEC, FD_POLICY, USE_DENSE, SLIDE_STAGES), dev/attn/build_attn.py, dev/moe_wm/build.py.

## UPDATE 2026-10-26: wgmma verify attention (dev/wg)
wgmma kernel (Q/K fp8 core-layout smem, V converted to f16 MN-major, descriptors LBO=1024/SBO=128 with tnspB=1) is CORRECT (relerr ~7e-4 vs v7) but only ~parity with FA3: 64-key tiles full verify 16/32 = 213/400us lab vs FA3 ~242/458 lab (official: 215.7/402.5 vs 205/391 stock => 0.95x). 128-key tiles no better (219/421). Ablation: V conversion ~70-110us, loads ~70us, softmax skeleton ~210us at B=32. Not promoted. Next idea if continued: overlap softmax with wgmma via 2 warpgroups ping-pong, cheaper V conversion.

## UPDATE 2026-10-26b: persistent MoE grid (dev/moe_wm, wm_set_persist, exp moe_wm_lab_persist)
Persistent mode (fixed grid of 132*k CTAs looping over (item,tile) units) is correct (stress + check, persist=2) but NOT faster: us per M (k0=baseline, k4=best persistent): M1 46.3/45.1, M8 94.0/94.7, M16 79.1/74.8, M32 180/186, M64 228/233, M128 260/283. Only M=16 gains ~5%. Not promoted; the per-item grid with early exits is already near the best. wm_cuda_v18pre.py is the pre-change backup.

## UPDATE 2026-10-26c: lossless exponent-compressed BF16 weights (dev/dense, exp bf16_comp_lab1)
Built `dense_crepack` (4-bit exponent code = emax-exp, sign|mant byte, 16 contiguous exps, ~1e-4 escapes fixed up exactly from a sparse list in the epilogue; 0.75x bytes) + `dense_ckernel`/`dense_crun`/`dense_cgemm`. Bit-exact on one-hot probes incl. escapes, relerr 1.6e-3 (= bf16 output rounding) on all shapes. H200 lab (graph timing incl. fixed overhead, us): compressed beats uncompressed dense only at m<=8 (pooler m1 25.5 vs 28.0, m8 28.6 vs 30.2; qkv m1 19.1 vs 19.5), loses at m>=16 (mma.sync token loop + decode ALU) and my dense kernels lose to cuBLAS for m>=16 anyway. Only 3-4 of 30 cases could gain -> est. +1-3% total. NOT promoted. A compressed GEMV for m=1/8 (hand-tuned to ~4TB/s on 0.75x bytes) is the only remaining use.

## UPDATE 2026-10-26d: new bests (all public-submitted via tools/promote.py)
- **Attention 1.522x** (`attn_fd3_pdl2`, was 1.394): KEY INSIGHT: the judge's page table is a random permutation (page=1) so decode attention is *DRAM random-access bound* (fd2 loads-only ablation = full time). Fetching whole 128 B rows (both KV heads in one CTA, `fd3_partials`, `dev/attn/attn_cuda.py`) costs ~174us vs ~280us for two 64 B half-row passes (gather microbench in lab_attn3.py: 128B_row 3.7TB/s vs 64B 2.1TB/s). fd3 = decode full B>=2 (policy variant 3); B=1 full + all sliding stay fd2. Plus PDL chain prepare->fd->merge only at batch 1 (Triton `launch_pdl`, gdc_wait/gdc_launch_dependents work on the judge; PDL at B>=16 hurt sliding by ~5us).
- **Decoder 1.481x** (`decoder_b5_rs`): fd3 attention + BF16-v5 dense paths (FP32 gemv at 1 token, transposed split-K pooler <=16 tokens; `USE_B5=1` in dev/build_decoder_all.py) + row-split MoE for >32 tokens.
- **MoE 1.859x** (`moe_wm_v18b`): new `wm_gemm_rs` (ring=4, `(4,4,4)` for M>32): the 4 warps own different weight rows, x tiles staged once per CTA through smem (no cross-warp K reduction, 4x less x L2 traffic). verify 256: 376->313us, verify 128: 250->243us. M<=32 keeps (4,4,1)/(2,4,0): decode cases are already at the HBM floor (~28 MB per active expert, 3.7 TB/s).
- Dead ends this session: persistent MoE grid (no gain), lossless exponent-compressed bf16 dense (gain only at m<=8, not promoted), wgmma verify attention (parity with FA3), fd3v/fd3w verify kernels with both kv heads (issue/latency-bound, 420-530us vs FA3 ~460 at B=32; mma.sync e4m3 is emulated on sm_90 -> use f16 mma for QK).
- Floors: MoE real HBM floor at 128/256 tokens is ~162/214us (not the harness SOL 101/109), so MoE max remaining gain ~+5%.
- Leaderboard before these: moe #225 (1.796), decoder #378 (1.36), attention #469 (1.39), bf16 #652 (1.174). rank-50 thresholds were moe 2.15, decoder 1.91, attention 2.25, bf16 1.50.

## UPDATE 2026-10-26e: BF16 dense2 (smem x + cluster DSMEM K-split, optional exponent-compressed weights) - NOT promoted
`dense2_kernel`/`dense2c_kernel` in dev/dense/dense_cuda.py (lab: dev/dense/lab_bf16e.py; exps bf16_dense2_lab1b/lab2). Correct (bad=0) on all shapes. Lab-scale times vs cuBLAS: qkv m8 20.1 (19.1 compressed) vs 21.1; pooler m8 29.6 (27.2 compressed) vs 30.3; unpooler m8 10.8 vs 11.8; at m>=16 it only ties or loses (o: 13.8 vs 12.1; pooler m32 42 vs 31). Ablation of the old dense kernel showed the cost was NOT weight loads or mma but x re-reads from L2 (every N-tile CTA reads the whole activation) + skeleton; dense2 fixes the x traffic but still sits ~4us above the pure-read floor (probe: 38 MB in 17.2us lab-scale, tiny graph 9.8us). Gains only at m<=8 (3 of 30 cases) -> not worth shipping. Leaders at 1.5x on this board must be doing something we are not (compression + a near-floor kernel?).

## UPDATE 2026-10-26f: SERVING PATCH (main event; track opens Nov 5 noon PT, `okc serve check` refuses before)
State of `serving/sglang` working tree (`okc serve diff` -> serving/patch.diff, ~61 KB, 6 files):
- NEW `python/sglang/srt/layers/muse_spark_wm_experts.py` (GENERATED by `dev/serving/gen_module.py` from `dev/moe_wm/wm_cuda.py` + `conv_cuda.py` + `dev/serving/module_template.py`; regenerate after any kernel change, never hand-edit): weight-major NVFP4 expert kernels (route, gate_up+silu, down) built with `torch.utils.cpp_extension.load_inline` (compiles in ~40 s, sm_90 only) + raw->wm and wm->Marlin layout conversion kernels. `WM_MAX_TOKENS = 24576` (wm beats stock Marlin at EVERY batch size on the H200: M=1024 593 vs 1092us, 4096 2.5 vs 5.3ms, 8192 5.0 vs 11.0ms; decode/verify 1.4-1.9x), conversion to a shared Marlin scratch is only a rare fallback (2.5 ms/layer, exact: the converted tensors equal SGLang's own Marlin tensors bit for bit, verified on the H200 incl. scales and global scales).
- `quantization/muse_spark_nvfp4_moe.py::_process_weights_for_marlin`: if `wmx.supports(layer)` (SM90, 32 local of 256 experts, 4096x4096, top8) builds the wm layout from the raw HF params (try/except -> falls back to stock Marlin on any failure) and drops the raw params (no second copy of the experts => no extra memory).
- `muse_spark_fp4_moe.py::_marlin_expert_outputs`: top-of-function hook: wm path for 1..WM_MAX_TOKENS tokens (same return contract: [m, top_k, H] + local ids with -1 for routes outside this GPU), else rebuilds Marlin tensors in the scratch. Plus earlier draft changes (fused hi/lo router for 8<M<=256, small-batch fused post-expert norm, Marlin block_size_m=16 for 32..512 tokens, now unused for wm layers).
- Numerics: wm vs Marlin differ by ~6e-3 rel L2 (each is ~4.5e-3 from exact); same gs rounding as Marlin (bf16 gs*2^119 gate column).
- Verified so far: (1) conversion kernels vs torch replicas locally; (2) real H200 lab (`dev/serving/lab_conv*.py`, exps serving_conv_lab1/2): module compiles, wm->Marlin tensors identical to the real ones, wm vs Marlin timings; (3) local RTX3050 stub harness `dev/serving/test_layer_stub.py` runs the REAL patched MuseSpark1xFp4Experts control flow (stubbed SGLang deps) vs reference: all batch sizes pass. The okc submission checker REJECTS submissions that edit loaded modules, so the real classes cannot be exercised in a kernel-problem lab.
- NOT validated: anything inside the real server (CUDA-graph capture, DFlash verify batches, 8-rank concurrent compile, load-time memory). First thing on Nov 5: `okc serve check patch.diff` and read its log; if the wm path crashes the stock Marlin fallback only triggers for build failures at load, not for runtime errors.
- Ideas queued: token tile of 64-128 for prefill-size batches (NCH 8/16 in the row-split kernel), speed up wm_to_marlin (60 ops/word), PDL/fusion of route+norm, fd3 attention is irrelevant (verify uses FA3 q_len 8).
- 2026-10-26g: Marlin rounds the fp32 accumulator to bf16 BEFORE multiplying by the bf16 global scale (marlin_template.h write(): `res = float2num(c); res = __hmul2(res, global_scale)`); the wm epilogues now mirror this (bfr(bfr(acc*256)*gs)), so wm vs stock Marlin differ by only ~5e-4 rel L2 (was 5.9e-3) - important for "answers must match stock's". Kernel-problem error vs fp64 reference rises slightly to ~5.2e-3 (same as Marlin). `dev/moe_wm/wm_cuda.py` carries this; earlier archived submissions (moe_wm_v18b, decoder_b5_rs) do not.
- Backups: serving/patch_wm_v1.diff (full, == patch.diff) and serving/patch_safe_v1.diff (conservative: router/post-norm/block_size_m only, no weight-major experts). Submit the safe one if the wm check crashes, then debug wm from the check log.

## UPDATE 2026-10-26h: wgmma big-M MoE negative result
- wgmma weight-major MoE (WM_WG, ring==7, wm_permx kernel) is bit-exact vs row-split but 1.5x slower (3862 vs 2513us at M=4096). Not in the serving module (gen_module strips the dispatch line; compile arch back to sm_90). Needs a producer/consumer warp-specialised redesign (decoded bf16 A in smem) to be worth it.
- Serving module regenerated; test_layer_stub passes bs 1..257 (rel L2 ~5.5e-3). serving/patch.diff refreshed (okc is at ~/.okc/bin/okc; `okc serve diff serving` writes serving/patch.diff itself).
