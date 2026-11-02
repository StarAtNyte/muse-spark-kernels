# KPERM3 + SC6 WGS MoE stack, v3

V2 passed CUDA compilation but its official check failed during `prepare()` because the extracted candidate used `_time.time()` for progress logging without importing `_time`. V3 restores `import time as _time`; all kernel code and dispatch are otherwise byte-identical to v2.

Candidate SHA-256: `0d5441bbd426c53ce3432385564cdb9bd3319beaba9e854395dfffa60c96d504`. Source parses successfully.

Official check `60769e4d692d4e9c` has been received by the judge. Await correctness and timing; v2 failed before timing.
