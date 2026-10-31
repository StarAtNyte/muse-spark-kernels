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

Run `ap-nXTmnofmapKExRykAtEDKB` completed with return code 0. All 12
configurations passed the driver's finite and relative-output checks; relative
L2 versus the incumbent ranged from 0.00319 to 0.00674. Every stream-K timing
was slower. Median paired parent/candidate ratios:

| Context | Grid 132 | Grid 264 | Grid 528 |
| ---: | --- | --- | --- |
| 32768 | 0.9774 / 0.9765 | 0.9780 / 0.9774 | 0.9711 / 0.9707 |
| 65536 | 0.9845 / 0.9851 | 0.9848 / 0.9846 | 0.9798 / 0.9802 |

Each cell lists seeds 539 and 1427. Retain the normal route and do not submit
this stream-K implementation on these results. The screen does not isolate
prefix setup from merge costs or rule out a different persistent scheduler.
