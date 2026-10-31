# Long prefill stream-K screen

The current scored prefill source (SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`)
contains a stream-K attention kernel but leaves it disabled. It assigns
contiguous key-tile work to a fixed CTA grid and merges partial results; the
normal route assigns each query block to a CTA. With causal attention, later
query blocks have longer key ranges, so more even tile allocation may reduce
the final wave's tail. The per-CTA prefix construction, partial writes and
merge may instead outweigh that gain.

The public-image H200 driver compares normal variant 43 against stream-K grids
of 132, 264 and 528 CTAs on synthetic B=1, query length 16384, contexts 32768
and 65536, page size 1 and two seeds. It checks finite values and relative L2
against the incumbent, and measures six alternating cold-L2 graph replays.
The subprocess has a 300-second limit; raw results and failures are in
`modal.log`. This screen is not an independent correctness check or full-layer
score; a promising candidate needs both before promotion.
