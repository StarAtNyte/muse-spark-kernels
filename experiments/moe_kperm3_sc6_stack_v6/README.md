# KPERM3 + SC6 WGS MoE stack, v6

V6 fixes the main-path dispatch: when `prepare()` validates the exact lossless SC6 scale encoding, `kernel_fn()` now selects the compiled WGS module carrying `-DOKC_SC6W=1` (`state["sfxm"]`). Tiny/MID paths still use their own `state["sfxs"]` module. This is the first candidate in this lineage that times the intended SC6 WGS kernel.

Source SHA-256: `8e004f446e1611fc9ff039eb17eb5f8af0d3872340574f61c27ec21da50538a1`.

The predecessor's pybind-export check `01fe098507f94a04` is queued and only validates the setup fix; no performance result is available. V6 official check `daacb84374274ef2` has been accepted and is queued. This is the first official candidate in this lineage whose timed main path selects SC6 WGS.
