# Decoder warp-parallel WGS prefix

## Hypothesis

The decoder's WGS path computes expert-cost prefixes serially in one lane before launching stream-K work. Parallelizing that prefix over the existing producer warp should shorten the schedule prologue, especially at larger batches, while retaining the exact incumbent cost model and arithmetic.

## Candidate lineage

- Problem: `muse_decode_layer`
- Parent: `experiments/scored_decoder_compact_qkv_o_v1/submission.py`
- Parent SHA-256: `9e7c8b73b0471d04f4f0cf451bea6ad372d7356e3103e7f57d7253df234aebb2`
- Candidate: `candidate.py`
- Candidate SHA-256: `af7620681cca64b123b28d79ee17b56da4109b631ed8288ea5e62cd42cf865c1`
- Scope: replace serial WGS cost-prefix/range computation in both embedded decoder MoE source variants; preserve cost table, unit weights, math, and other kernels.

## Paired H200 screen

Exact-parent judge-image screen; sliding decode and verify, batches 1/16/32, two seeds each. All 12 candidate and parent correctness checks passed.

| Batch | q_len 1 speedups | q_len 8 speedups | Batch geometric mean |
|---:|---:|---:|---:|
| 1 | 1.0000×, 0.9988× | 0.9940×, 0.9952× | 0.9970× |
| 16 | 1.0006×, 0.9942× | 1.0070×, 1.0049× | 1.0017× |
| 32 | 1.0086×, 1.0091× | 1.0025×, 0.9978× | 1.0045× |

Overall geometric mean: **1.0010449×**, about **+0.10%**. This is a promising but small screen result, with regressions at batch 1 and gains concentrated at larger batches. The full official decoder workload scored 2.20152× (95% CI 2.16850–2.23455×), below the 2.26032× incumbent. It passed 13/13 correctness cases, but the scored result does not support promotion; the paired screen did not predict the broader full-layer result. The incumbent remains promoted.

Raw timings and Modal run log: `modal_judgeimage.log`. The technique was inspired by the public decoder notes in [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge); their reported gains are not assumed to transfer to this implementation.

## Expanded paired screen: full and sliding

After the official result, repeated parent/candidate timing over all 12 combinations of layer (full/sliding), batch (1/16/32), and q_len (1/8) with one seed and eight alternating paired replays. All 24 measurements passed correctness. Aggregate candidate/parent geometric mean was **1.00225×** (+0.23%); sliding was **1.00587×**, full attention **0.99864×**. Full B1 regressed 0.5–1.5%, while B32 sliding decode gained 1.49%. Raw log: `full_modal_judgeimage.log`; driver: `full_pair_driver.py`. One seed and a low aggregate effect are insufficient to override the lower official score.

## Decision

Do not promote yet. Official scored job `03083a8a606d42e2` measured 2.20152× (95% CI 2.16850–2.23455×), below the current 2.26032× incumbent, job `b72f02c7eb8745f5`. All 13 correctness cases passed. The official per-case profile is under `profiles/`. The broader paired screen is only +0.23% and the official draw is lower than the incumbent; the candidate remains archived. Repeat paired seeds before reconsidering. The adaptive v2 and router candidate remain untested.
