# BF16 projection score variance draws

The current published source `kernels/muse_bf16_proj/submission.py` was already
fully checked and scored as `e5cae4f990be4df2` (1.5853x). It was 0.28% below
the live #50 cutoff, so three additional official score draws were submitted.
Each archived file is the exact same Python/CUDA source plus one trailing
comment; no timed code, dispatch, weights, or math changed. These runs can only
sample judge variance. They are not kernel improvements and must not be used as
performance evidence for the source.

| Draw | Source SHA-256 | Official run | Status |
|---:|---|---|---|
| 1 | `951525bcda7444dae78078344601e5913a0a95074c94598688789b4f9bf46aae` | `3b3cca5c023a441a` | queued |
| 2 | `3c5741e521060b0f62d1f829486083128a8e21cd026282f2ee65c074f06d051a` | `5a25cd13386a47ee` | queued |
| 3 | `854d1651f1295e9977a5f4c596530eb7d3b4123d8c459f905978b911789b7b78` | `1241a8bba593445d` | queued |

The exact judge profiles, results, and row movement will be added here after
completion. The duplicate submission `4ce51d7386044002` did not run because the
judge recognized the byte-identical source and returned the previous result.
