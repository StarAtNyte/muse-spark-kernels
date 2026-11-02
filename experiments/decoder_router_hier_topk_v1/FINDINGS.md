# Fused router hierarchical top-k

Prepared exact hierarchy for the decoder's fused Triton route. The existing code packs the monotonic float score together with reverse expert ID, so every expert has a distinct key and ties match the existing lowest-ID rule. Any global top-8 element must appear in the top-8 of its 64-element partition; selecting 8 per partition and then top-8 from those 32 is exact.

The four-partition layout is intended to align with `_sl_tile_route`'s existing four-warp launch. The candidate changes only the `MODE == 0` top-k inside `_sl_tile_route`; all other routing, weights, math, and kernels stay byte-for-byte as in the WGS-v1 candidate. No timing or correctness claim yet.

A recent [vLLM Muse routing kernel PR](https://github.com/vllm-project/vllm/pull/64553) uses warp-local candidate ranking before global selection. Its `redux.sync.max.f32` path is for Blackwell SM100, not H200 SM90, so it is deliberately excluded.

The fused route is only about 2 us in existing decoder profiles, much smaller than the integrated MoE kernel. This is a bounded secondary optimization. Run the paired full-layer H200 screen only after official job `03083a8a606d42e2` stops occupying the judge queue.
