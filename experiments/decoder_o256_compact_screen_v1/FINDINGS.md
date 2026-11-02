# T=256 compact output projection: rejected

Extended the T=128 compact QKV/O candidate with the same O-layer family at
T=256. The paired screen ran on the challenge's public judge image with an
H200. It covered full and sliding verify-32, three seeds, and 24 alternating
CUDA-graph replays per implementation and seed. All six parent/candidate
outputs passed the reference checks.

Candidate/parent speed ratios were 0.9956x, 0.9981x, and 1.0004x for full
verify-32, and 0.9947x, 0.9965x, and 0.9957x for sliding verify-32. Aggregate
parent/candidate was 0.9968x: the T=256 route is about 0.32% slower overall,
with the consistent regressions on sliding verify-32. Reject this route and
keep the previous candidate limited to T=128.

The parent remains the archived T=128 QKV/O stack. The new source is preserved
as `submission.py`; the exact timings and command are in `paired_screen.json`.
This experiment spent no official checks or scored-run quota. The live public
decoder score/rank are unchanged at 2.2603x / #114.
