# Fused prefill MoE queue barrier hypothesis

The exact public-best parent is SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`.
Short-prefill NCU counters showed substantial aggregate wait/barrier stalls,
but do not attribute them to a particular instruction. This candidate tests
one specific extra consumer barrier immediately after reading the producer's
two-slot work queue.

With `MOE_ALLLANE_RELEASE=1`, the empty mbarrier is initialized to 256 and
each consumer reads `iring[k]` before its own arrival. The producer waits on
that barrier before reusing the slot. Hypothesis: these individual arrivals
already prevent slot reuse before every consumer's read, so the additional
256-thread `bar.sync` can be omitted for this mode. The original barrier is
retained for the alternative one-arrival mode. The proof also depends on the
publication/acquire ordering, all consumers reading the same item, matching
barrier phases, subsequent warpgroup synchronization and termination behavior;
these must be checked on GPU and under race/synchronization tools. Passing
output tests alone does not prove race freedom.

Candidate SHA-256
`2448601e2bf24b7b150000998ab30ced06a2adf72078ae8bdef01a304f2c3ae0`.
The H200 public-image driver applies only this CUDA change, shares prepared
weights and plans, checks both full-layer outputs against the task reference,
and compares 16 alternating cold-L2 graph replays on three representative
prefill shapes. Attention is pinned because the local SGLang stub prevents
its tuner from running. The subprocess is limited to 300 seconds; raw output
and any failure are in `modal.log`. This is an exploratory screen, not an
official result. No submission or promotion without further race validation
and the complete official check.
