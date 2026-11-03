# Corrected compact QKV transfer

Adds the missing module-level `torch` import to v2, which already retained fresh-thread CUDA context binding. A three-seed QKV-128 component screen measured 1.085–1.095× and passed reference checks; see `../projection_compactq128_paired_v1/FINDINGS.md`. Private full check `d96792184a114eef` is pending. Submission-wide source and precision audit is still required before a public run.
