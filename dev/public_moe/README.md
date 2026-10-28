# MoE technique stack

MIT upstream team bigboateng/operatorstack/open-kernel-challenge-solutions, commit6d9b77656b6c295e421be9e1424b0f8f99299b0a. Original source SHA281d56e94d22f7ebb440274d25c240697b4728af2586da252fb1befb1f35cc35. LICENSE and NOTICE preserved; SGLang-derived portions remain Apache2.0. Only a first-line problem marker and license/source comments were added; executable code unchanged.

Independent official check moe_operatorstack_v1 /416e25689b8e4426 passed5tests/6timings at1.846357x, below our aggregate1.933478x. Individual baseline-normalized speedups show public M32 1.452x vsour1.349x and M1282.123x vsour2.051x; public loses M1/M8/M16/M256. These are different fresh-input runs, so shape selection requires a complete combined check.

hybrid_m32_m128_v1.py namespaces upstream, prepares both lossless layouts, selects public only M32/M128 under identical expert dimensions/local bank, retains our validated count-tiered/native implementation and fallbacks elsewhere. Extra prepared weight copies and setup time are a cost, not hidden input work; all activation work remains timed. Private experiment moe_hybrid_m32_m128_v1 is pending. build_hybrid.py reproduces it. No promotion or score gain claimed yet.

First hybrid check434fd07a6cd048d0 failed a namespace integration bug on M128 skewed: JIT argumentBM renamed to_om_BM but keyword calls were not renamed. No score. build_hybrid.py now renames keyword arguments consistently and produces hybrid_m32_m128_v2.py; official check434b1d6c0336443e pending.

Corrected hybrid434b1d6c0336443e passed5tests/6timings at1.9720838702x(CI1.942503–2.001665), improving aggregate overour1.933478x. Promoted and publicjoba4b30b80345f499f launched. Source aliases and JIT keyword names are consistent.

Public hybrid a4b30b80345f499f DONE1.9697822631x(CI1.940236–1.999329), correctness passed, versus previous public1.9336267768x.

## Whole-decoder integration

build_decoder.py creates decoder_moe_stack_v1.py from immutable public decoder_stack_v3, preserving its projection/router/attention/boundaries and adding the exact namespaced public MoE component from the validated standalone hybrid. Maps decoder's post_expert_norm to standalone post_norm_w for preparation and uses the existing effective post_expert_w tensor in the new path. Decoder routes already shifted/modulo to the local bank become local-start0 expert IDs for this component. Only M32/M128 select it; original paths remain elsewhere. Fixed native CUDA strings are losslessly compressed to fit964685bytes; no input data is cached or compressed.

Official private check2dca3780329f4907 (decoder_moe_stack_v1) is running. Do not infer decoder gains from standalone MoE timing. New source is experimental until all13tests/12timings pass and aggregate improves.

Whole decoder result2dca3780329f4907 completed:13tests/12timings all pass,1.7061543707x(CI1.680562–1.731747), below existing decoderprivate1.707723x/public1.715586x. Do not promote. Baseline-normalized per-case comparison: decode32sliding0.975x/full0.999x, verify16sliding1.008x/full1.011x. Different routing/activation distributions matter; standalone expert gains do not establish a whole-layer gain. Keep existing production decoder.

M32-only decoder stack (candidate SHA15cfa49a6fb187647aa597a2d30b9d3bf39d2a38c2f424e18e928a8fb6196c4f) also passed13/13 official checks at1.702752x (CI1.677211–1.728294), below decoder_stack_v3 at1.707723x with overlapping intervals. Restricting public MoE dispatch to its putative winning 32-row regime did not produce an end-to-end improvement; do not promote. Rebuild with `python dev/public_moe/build_decoder.py --m32-only --base experiments/decoder_stack_v3/submission.py --out dev/public_moe/decoder_moe_m32_stack_v1.py`.
