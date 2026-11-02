# KZ projection routes, isolated v4

The repaired v3 check failed before scoring when the helper eagerly launched its M=128 `o` kernel. V4 disables that eager warm launch (`OL_CFG = {}`) and removes `(32, "o")` from the KZ dispatch set. This leaves the selected QKV, router, pooler, and unpooler routes intact and falls back to the incumbent for `o`. The embedded module and candidate both pass Python AST parsing.

Candidate SHA-256: `76e42224d8a8c9dfcbb943e2a812d3996dbdc9867e2181e7675daaa3516a3d3a`. Embedded module SHA-256: `d50b4373ee55e5363ad34788533c97b4be37be22c17069a7d3c5c18036ffcb5c`.

Official check `9f3f921904df4ab4` has been received by the judge; timing/correctness result pending.
