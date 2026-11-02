# KPERM3 + SC6 WGS MoE stack, v5

V5 exports the already-defined C++ `okc_attrs()` diagnostic through `load_inline`'s pybind function list, fixing v4's setup failure. Source audit then found that `kernel_fn` dispatches the main path with `state["sfx"]` rather than `state["sfxm"]`; this means the check does **not** exercise SC6 WGS in its timed path. The queued check validates the export/setup repair only; its timing is not evidence for SC6. V6 fixes the dispatch. Parent candidate: `moe_kperm3_sc6_stack_v4`.

Source SHA-256: `80d07b2d2034661317e0d853cd470a7b77d68ac8b199e415f7b84286ca8a91c2`.

Official check `01fe098507f94a04` was accepted and is queued. The first CLI invocation omitted the explicit problem because this research candidate has a Python import before its `# okc` marker; it was rejected locally without creating a judge job. The second invocation named `muse_moe_w4a16` explicitly.
