# QKV-32 paired backend comparison

On H200 with the public judge image, prepared only the required chenwainuo parent and vjeux candidate backends and timed qkv m=32 with cold-L2 alternating CUDA-graph replays. Both outputs passed reference checks for all three seeds:

| Seed | Parent µs | Candidate µs | Speedup |
|---:|---:|---:|---:|
| 61 | 18.816 | 15.248 | 1.2340× |
| 62 | 18.832 | 15.232 | 1.2363× |
| 63 | 18.656 | 15.328 | 1.2171× |

These event-timed values include harness overhead; the judge's private run measured 14.42 vs 10.74 µs in separate parent/candidate runs. The timing screen targeted a single shape, not the overall score. Private candidate check `e4df402a11e94f52` passed 9/9 at 1.54234×; the parent check scored 1.53642×. The selected candidate kernel is SK2 and its split reduction uses FP32. Other routes of the full submission still need precision and source review before any public submission. Driver, raw log and exact candidate source (`../projection_fp32_qkv32_sk2_v1/submission.py`) are retained.
