# Independent Forge attention judge

`dev/forgekernel/attention_judge/` contains an independent mathematical page-1 sliding attention judge and generated input groups, including returned mutated K/V caches. The first draft incorrectly rounded queries to FP8 and produced 3.84% relative L2 error; its source and log are retained as `fp8_query_judge.py` and `fp8_query.log`.

The corrected BF16 query path was compared to the challenge reference on H200 for three preliminary cases (batch/length 1/1, 1/8, 2/1): output relative L2 errors were 0.00169, 0.00164 and 0.00165, and the newly written K and V values matched exactly. See `bf16_query.log` and `driver.py`. This is not full coverage or the official challenge tolerance.

The manually supplied judge passed Forge's static validation in run `.artifacts/muse_attention_sliding_261031-164209-d8c3`; calibration job `1793444260` is pending. It compares the judge with the recorded best optimized source on all configured cases. Only a successful, reviewed calibration can support running the focused optimization search; this page-1 task cannot replace the official full attention suite. The task's shapes now expose a `tokens` axis to express flattened cache locations.
