# Short-prefill MoE mixed tile ordering

The scored source (SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`)
orders all larger expert tiles before smaller tiles. This candidate selects the
existing order 2 for token counts below 8192, mixing most 32/64-row tiles among
larger ones. The hypothesis is that heterogeneous work reduces producer/consumer
and dependency tails in the 753-token cases; the reordering can also degrade
cache locality or tensor utilization. Long-prefill ordering is unchanged.

Candidate SHA-256
`f2fc3bc04403a6cf5981e57ce77817e5e7f0c9c71f69342d8833ee6f24a89eb0`.
The bounded H200 public-image driver changes only the host-side order, shares
prepared weights and plans, checks each output against the task reference, and
compares 16 alternating cold-L2 graph replays on B=1 sliding/full and B=4
sliding, query length 753. It pins attention because the SGLang stub cannot
run the attention tuner. One seed per case and local checks do not establish an
official score; full official correctness is required before promotion. Raw
results and failures are saved in `modal.log`.

Run `ap-pW13L402n4Rysb728HVnuL` completed with return code 0. Both variants
passed each reference screen. Parent/candidate full-layer medians were
513.4/512.1 µs for B=1 sliding, 885.2/881.3 µs for B=1 full, and
1644.8/1647.3 µs for B=4 sliding. These one-seed, sub-0.5% mixed results are
too small to establish a gain. A multi-seed paired-ratio repeat of B=1 sliding
is running in `../prefill_moe_small_order2_repeat_v1/`; retain the incumbent
pending stronger evidence.
