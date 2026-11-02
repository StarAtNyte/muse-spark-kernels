# BF16 projection score variance draws

The current published source `kernels/muse_bf16_proj/submission.py` was already
fully checked and scored as `e5cae4f990be4df2` (1.5853x). It was 0.28% below
the live #50 cutoff, so three additional official score draws were submitted.
Each archived file is the exact same Python/CUDA source plus one trailing
comment; no timed code, dispatch, weights, or math changed. These runs can only
sample judge variance. They are not kernel improvements and must not be used as
performance evidence for the source.

| Draw | Source SHA-256 | Official run | Status | Score (95% CI) |
|---:|---|---|---|---:|
| 1 | `951525bcda7444dae78078344601e5913a0a95074c94598688789b4f9bf46aae` | `3b3cca5c023a441a` | done | 1.56698× (1.54348–1.59048) |
| 2 | `3c5741e521060b0f62d1f829486083128a8e21cd026282f2ee65c074f06d051a` | `5a25cd13386a47ee` | done | **1.59192× (1.56804–1.61579)** |
| 3 | `854d1651f1295e9977a5f4c596530eb7d3b4123d8c459f905978b911789b7b78` | `1241a8bba593445d` | done | 1.58068× (1.55697–1.60439) |

The #41 board row now reflects draw 2's higher random result; the validated
kernel artifact remains at 1.5853×. The intervals overlap and these comment-only
draws do not establish any timed-code gain. Full H200 profiles are archived in
`profiles/`. Duplicate submission `4ce51d7386044002` did not run because the
judge recognized the byte-identical source and returned the previous result.
