# Compact T=128 QKV + O routes stacked into the current decoder

Date: 2026-11-02. Candidate SHA-256:
`aa354bd2472393cfb407a50260dbc25543dab3763c4b45bcffca4f450ca98b6e`.
Parent source is the current decoder in `kernels/muse_decode_layer/submission.py`
(SHA-256 `82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19`).

## Change

Grafted the previously measured compact QKV and O projection routes from our
archived decoder winner onto the current decoder. They activate only when
`T=128` (batch 16, q_len 8); every other timed shape falls through to the
parent implementation. Preparation retains the parent’s routes and adds the
packed QKV/O weights and kernels.

## Paired judge-image H200 screen

Three input seeds per mode, 24 alternating graph replays per implementation
and seed. Parent and candidate passed the reference check for all six rows.

| Mode | Seed | Parent µs | Candidate µs | Parent/candidate |
|---|---:|---:|---:|---:|
| Full | 4001 | 315.3 | 314.9 | 1.0014× |
| Full | 4002 | 341.8 | 341.0 | 1.0022× |
| Full | 4003 | 327.2 | 326.4 | 1.0025× |
| Sliding | 4001 | 249.6 | 246.9 | 1.0111× |
| Sliding | 4002 | 251.0 | 251.1 | 0.9997× |
| Sliding | 4003 | 225.8 | 225.6 | 1.0008× |

Geometric candidate/parent gain: **1.00294× overall**, **1.00203× full**,
and **1.00385× sliding**. The gain is small; the largest positive row is one
sliding seed. Do not interpret this screen alone as a robust speedup.

The generic CUDA Modal image could not import `tvm_ffi`; that attempt ended
before preparation or timing. The screen was rerun on
`lmsysorg/sglang:dev-muse-spark-open`, the same judge image used by the other
whole-decoder paired screens. H200 run:
https://modal.com/apps/baptinjean20/main/ap-dyFKhVLU3X7PHRuzjZiCzG

## Official full-workload check

Job `d3ec1a729a6c4107` passed **13/13** and measured **2.271×**, 9,962.7 µs,
with 95% CI **2.236–2.305×**. This is above the published 2.26032× point
estimate, but the intervals overlap. It is an off-board check, so the live
score and rank remain **2.2603× / #115**. Current top-50 line is about 2.310×;
this candidate has not crossed it. Preserve for the next decoder scoring quota,
but do not label it a confirmed leaderboard gain.

The full check profile, JSON and log are in `profiles/`. Source and paired
screen driver are preserved alongside this note.
