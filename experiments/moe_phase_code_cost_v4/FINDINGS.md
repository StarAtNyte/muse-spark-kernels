# MoE cold-vs-warm per phase, read flush (2026-11-01) - closes the diagnostic

History: v1/v2 crashed at bs=32 and v3 used `flush.add_(1)` (dirty L2: write-backs inflate "cold" by a near-constant 4-7 us at
every size). The judge flush is a read ("reads a buffer larger than L2", TASK.md), so v4 uses `flush.sum()` and the file's own
`kernel_fn` (the crash was my hand-built `okc_run` call, not the kernel) with `-DOKC_DEBUG=1` patched into the build tuples.

Median CTA time, cold minus warm code (cold data in both):
- bs=8: final -0.5 us; bs=16: +1.3 us (1.4%); bs=128: -0.3 us;
- bs=32 (WGS): routed 1.8 us and gathered/joined 4.4 us identical cold and warm; barrier 113.5 vs 106.5 (+7.0), final 109.6 vs
  105.6 (+4.0). The 4-7 us (4-6%) accrues inside the streaming phases, not in the prologue.
Reading: first-pass instruction fetch in the WGS consumer/epilogue code is the likely cause (hidden at bs>=128 where the kernel
is bandwidth-bound, visible at bs=32 where it is latency-bound), but this is an inference. A dry-run warm-up has no idle
window: consumers gather x until 4.4 us and the first weight stage arrives about then. Prize: ~4-7 us of ~122 us on the 32-token
MoE load = ~+0.9-1.4% MoE board, ~+0.5% decoder. Not built.
