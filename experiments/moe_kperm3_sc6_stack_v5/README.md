# KPERM3 + SC6 WGS MoE stack, v5

V5 fixes a setup failure in v4 by exporting the already-defined C++ `okc_attrs()` diagnostic through `load_inline`'s pybind function list. No kernel math, dispatch, compilation flags, or timed path changed. Parent candidate: `moe_kperm3_sc6_stack_v4`.

Source SHA-256: `80d07b2d2034661317e0d853cd470a7b77d68ac8b199e415f7b84286ca8a91c2`.

Official check `01fe098507f94a04` was accepted and is in progress. The first CLI invocation omitted the explicit problem because this research candidate has a Python import before its `# okc` marker; it was rejected locally without creating a judge job. The second invocation named `muse_moe_w4a16` explicitly.
