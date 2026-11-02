# KPERM3 + SC6 WGS MoE stack, v4

V3 restored the missing `_time` module import, but its official check showed that `prepare()` also referenced `_t0` before initialization. V4 sets `_t0 = _time.time()` at the start of `prepare()`. Kernel math and dispatch are unchanged from v2.

Candidate SHA-256: `762a156e26231698282d0d7a3263ae98119a8f7c3a087c83d3702abd9e24321d`.

Official check `31f94f4fb0bd43e9` completed without timing the candidate. `prepare()` compiled the kernels, then failed in diagnostic setup because `_ext()` did not export the C++ `okc_attrs()` function to Python. No correctness or performance result was produced. The full judge status and failure log are archived here. V5 fixes the pybind export only.
