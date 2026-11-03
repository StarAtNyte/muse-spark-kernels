# Transfer Lee's warp-independent sliding decode path

This candidate starts from the current scored attention source `attention_warm_route_verify_v1` (SHA-256 `573f179558bb2eae13faa7a58ecfdd924196bb15e9bfd25fd552289ae903c1da`) and adds only the `swk` kernel plus its launch route for H200 batch-16/32, q_len=1, sliding attention. It leaves every other shape on our incumbent path.

The kernel assigns independent 16-key chunks to warps, issues each warp's K/V gathers together, computes a one-pass softmax, and combines partials within a small Hopper cluster. That removes the serialized key loop and the cross-CTA global split merge on these two shapes. Upstream's v84 notes report 2–4% lower latency there; v85's follow-up variants lost, so this exact route needs a paired result on our current parent.

The upstream implementation takes one KV scale, but the challenge provides separate `k_scale` and `v_scale` device tensors. This port reads both in the kernel: K scale applies to QK and new-key cache writes; V scale applies to new-value writes and the final weighted sum. It does not synchronize or inspect scale values on the host. The profiler hook was removed, and the route is shape/config-dispatched only.

Source SHA-256: `dcf940999ee4d754963dbe8bd757bcf9f3a68136403244a2eb58c217c9644c48`. Source attribution and the required Apache-2.0 license are in [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0).

The next gate is the complete official H200 check. If it passes, compare the exact scored result and per-case profile with our **2.6328×** public score before promoting or submitting.
