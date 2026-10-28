# FP8 probability/value math inside a warp-specialized pipeline

Starting component: attributed MIT bigboateng attention (source/license retained in candidate.py). Keep producer/consumer mbarrier pipeline and fused normalization/cache stores/merge. Adapt full-context Gluon verification only:

- P = exp2(score - running_max + 8), converted to E4M3 for PV. FP32 normalizer uses the same scaled P before conversion, so the constant factor256 cancels at normalization and split merging.
- V stays E4M3. Consumer transposes it into K-contiguous FP8 shared memory with NVMMASharedLayout(transposed=True), instead of converting to FP16.
- PV uses FP8 m64n64k32 with k_width4; new-key padding becomes32 to satisfy the K step. Every key and causal mask is retained.
- QK, FP32 softmax max/sum, output BF16 rounding, norm and gate are retained. This matches the precision allowed by the pinned FA3 FP8 path documented in research/fa3; all numerical checks remain mandatory.

This removes FP8-to-FP16 conversion and halves the converted V scratch. It does not assume a speedup: layout transpose cost, compiler scheduling, resource allocation and quantization error can offset the change.

Driver: ../modal/drv_gluon_fp8p.py. First independent correctness cases include short contexts, edges and scaled caches. Only passing variants proceed to six large-case alternating cold-L2 comparisons against the production attention hybrid. Official validation is required before promotion.

Environment triage: screen.log used Torch2.9's bundled Triton without mma_v2; screen_v2.log reports Torch2.11/Triton3.6, missing allocate_mbarrier. Neither reached a kernel correctness test. run_gluon_h200.py now pins Triton3.8.0 (available PyPI version verified). screen_v3.log failed importing a sibling runner module on the remote worker; corrected standalone runner embeds its own execution helper. Failed app ap-jpUZvKKXWOy5RUFwB4aNQV had zero containers and was explicitly stopped before retry. Current evidence screen_v4.log. One experimental GPU app at a time.

## Outcome

Trition3.8/Torch2.9.1 screen_v4 reached compilation: new-key NP32 was too short for64-byte transposed swizzle. screen_v5 uses32-byte swizzle for the new-key V tile only, retaining64-byte swizzle for long tiles. All9 reference checks passed (3 short/scaled/edge plus6 large cases). Full-case ratios B16~0.658–0.661 andB32~0.710–0.716 against production FP16 Gluon PV. Reject: layout/lowering/scheduling costs outweigh the datatype change in this implementation. No official quota spent. Final app ap-Cy1MtjjOD2T7rE44tEFAPK completed; no experimental GPU job remains live. Do not infer the exact bottleneck without a resource/profile comparison.

## Resource and counter comparison

resources.log completed: both versions128registers/zero spills; baseline86196sharedbytes vsFP878004bytes. PTX WGMMA instruction counts13vs9; wait sites4vs4. No register/spill explanation established.

profile.log completed focused NCU counters on both passing kernels. Baseline:156.19us, shared-store conflicts205979, storewavefronts2881416, MIOthrottle1.03%. FP8:238.02us, conflicts15528837, storewavefronts20951713, MIOthrottle15.27%. Shared-load conflicts265vs885 (small). These per-kernel counters point to store/layout traffic; they do not by themselves attribute every conflict to V rather than another compiler conversion.

make_vector_store.py therefore tries an explicit transposed register layout before V shared store, with16 contiguous K bytes/thread and128-byte swizzle for the128-key V tile. New-key tile remains32-byte swizzled. Arithmetic and normalization unchanged. Paired correctness/timing evidence vector_store.log; if it wins, confirm counters and official coverage.

## Counter-guided follow-ups

vector_store.log: all9 reference checks passed. Explicit transposed register layout plus128-byte V swizzle reduced the original FP8 regression: B16~188–190us versus originalFP8236–237us; B32~364–368us versus original455–457us. Still loses to the scored FP16 Gluon path (~156/314us), so no promotion.

shared_p.log: all9 reference checks passed. Supply FP8 P through a shared A descriptor, avoiding register DotOperandLayout redistribution; add proxy fence/consumer barrier before PV. It loses to vector_store (~199/380us), so reject this variant. This does not establish that all remaining conflicts are in P; further layout claims require attribution. Both follow-up GPU apps completed; no official quota used. Keep scored FP16 producer/consumer implementation as production.
