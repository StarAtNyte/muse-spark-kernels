# Long prefill KV-head interleaving

The scored prefill attention route launches query-block index as the fastest
grid dimension and KV head as the next dimension. For one split it therefore
runs one head's descending query blocks before the other head's. This screen
interleaves the two KV heads per descending query block, preserving one-to-one
coverage and longest-query-first order. The split route remains unchanged.
Better distribution may reduce tail, while changing cache locality and block
placement may cause a regression; no benefit is assumed.

The driver builds both variants from the exact scored source SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`.
It checks finite outputs and relative L2 against the incumbent and measures
12 alternating cold-L2 graph replays at B=1, query length 16384, full attention,
contexts 32768 and 65536, page size 1, two seeds. The H200 public-image runner
limits the subprocess to 300 seconds and stores raw results in `modal.log`.
This is a synthetic component screen; a promising candidate requires full
layer paired timing and official correctness before promotion.

Component run `ap-I2mvtgb7yndSPYjQqXX8Fb` passed its output comparison with
zero observed relative L2 difference. Median paired parent/candidate ratios
were 1.00565 / 1.00564 at context 32768 and 1.00267 / 1.00264 at context
65536, for seeds 539 / 1427. This small gain justifies a full-layer screen,
not a score claim. Raw output is `modal.log`.

`candidate.py` applies the interleaving only for long, full, one-split
attention; SHA-256
`8099523e1e9e97aba8ca51d6743f7331927548bc79bdf379a675635a27107099`.
`full_layer.py` compares it with the scored parent on full-attention contexts
0, 32768 and 65536, with pinned attention choices and alternating cold-L2
graphs. It compares layer outputs and routing against the incumbent, not the
independent reference. Run output is `full_layer.log`; the bounded runner
limits its subprocess to 300 seconds. Official correctness is still required.

`verify_mapping.py` exhaustively checked query-block counts 1–1024 for one,
two and four splits: the original and candidate formulas assign the same
unique `(query block, head, split)` tuples. This checks index coverage only,
not CUDA memory safety or attention numerics.

Full-layer run `ap-s5ms3Gx9p0btul5CxJaOym` completed. Routing IDs matched and
the hidden/stream relative differences were below 0.0001 in all three cases.
The median paired parent/candidate ratios were 1.00082, 1.00286 and 1.00169
at contexts 0, 32768 and 65536. There was substantial timing variation: ratios
of the separately reported medians differ from the medians of paired ratios,
and the two modules independently tuned some cuBLASLt plans. The SGLang stub
prevented attention auto-validation, so both choices were pinned to the same
variant. These are exploratory signals rather than reliable score estimates.
Raw output: `full_layer.log`.

Official private check `6a672f2beca74962` passed with no flags and scored
1.80145× (95% CI 1.77442–1.82847), versus the published 1.80923× (95% CI
1.78210–1.83637). The separate-run intervals overlap; they do not resolve the
small local gain. The exact checked hash was submitted as scored job
`e3127297ccc1451f`, which was running at this update. The incumbent remains
promoted until the scored result is available. Status and profiles are in
`../prefill_head_interleave_check_v1/`.

Scored job `e3127297ccc1451f` passed with no flags at **1.79978×** (95% CI
1.77278–1.82677), below the published best **1.80923×**. The intervals overlap,
so separate-run scores do not establish the small optimization's effect. The
public-best source remains canonical; scored status and full profiles are
archived with the check. No new leaderboard best is claimed.

`schedule_model.py` models each CTA's cost as its number of 128-key tiles,
with 132 identical workers greedily taking the next CTA. For query length
16384 it predicts separate/interleaved makespans of 672/671 tiles at context
0, 3499/3487 at 32768 and 6315/6303 at 65536. This is a simplified scheduling
illustration, not a GPU lower bound or proof of the measured cause: it ignores
cache, ramp, overlap, placement and variable per-tile execution time.
