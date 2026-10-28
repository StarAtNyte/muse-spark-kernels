# Infraax BN128 full-verify transfer screen

## Hypothesis

Infraax reported a BN128 / three-stage full-verify path beating its previous
BN64 kernel by 1.16x at batch 16 and 1.13x at batch 32. Test its complete
Apache-2.0 attention implementation directly against our current public
attention source on the same H200 and exact scored verify-full shapes.

## Method

Modal H200, CUDA graphs, 256 MiB read-only cache flush before each replay,
alternating baseline/candidate order, 20 paired replays per seed, three seeds
per batch. The official attention reference checker was run on each candidate
output. Modal app: `ap-1ZXV7BOecTyAFQoJPzGV50`.

## Results

| Batch | Seed | Our kernel (us) | Infraax kernel (us) | Our / candidate | Official reference |
|---:|---:|---:|---:|---:|---|
| 16 | 539 | 109.79 | 171.50 | 0.640x | pass |
| 16 | 1427 | 109.87 | 171.10 | 0.640x | pass |
| 16 | 7919 | 110.61 | 171.82 | 0.644x | pass |
| 32 | 539 | 193.71 | 330.38 | 0.587x | pass |
| 32 | 1427 | 191.49 | 330.22 | 0.580x | pass |
| 32 | 7919 | 191.89 | 329.52 | 0.580x | pass |

The candidate output differs from our output by relative L2 0.043–0.049, while
passing the official checker. It is substantially slower on all six paired
measurements. The published tile-size win was against Infraax's earlier
implementation and does not transfer to our already faster kernel. Reject;
do not spend an official check or alter the submission.

## Provenance

Source: `infraax/okc-kernel-challenge`, `kernels/muse_attention_decode/sonnet_v5.py`,
Apache-2.0. The exact candidate, license, Modal driver, and raw screen values
are retained here. Candidate SHA-256:
`9f90e9c40d9c208fa605a62008585941194c06c7853f6b325af96a4209efd3f1`.
