# KPERM3 + SC6 WGS MoE stack, v6

V6 fixes the main-path dispatch: when `prepare()` validates the exact lossless SC6 scale encoding, `kernel_fn()` now selects the compiled WGS module carrying `-DOKC_SC6W=1` (`state["sfxm"]`). Tiny/MID paths still use their own `state["sfxs"]` module. This is the first candidate in this lineage that times the intended SC6 WGS kernel.

Source SHA-256: `8e004f446e1611fc9ff039eb17eb5f8af0d3872340574f61c27ec21da50538a1`.

The predecessor's check `01fe098507f94a04` failed during C++ compilation because the diagnostic export was defined in the wrong translation unit. V6 inherited that defect; check `daacb84374274ef2` was received, but has no result yet. V7 removes the optional diagnostic binding while retaining this SC6 timed dispatch. V6 is the first candidate whose timed main path selects SC6 WGS, but its check cannot provide performance evidence until the binding issue is fixed.
