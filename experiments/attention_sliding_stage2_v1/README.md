# Sliding attention: two-stage split-KV pipeline

The best scored attention source uses three K/V stages for non-flat split-KV and sliding grids. Earlier `attention_stages_v1` screened depths four through six; this experiment tests two, which may shorten pipeline setup and reduce shared-memory use but could expose memory latency. The exact parent is `attention_plan16_v1/submission.py`, SHA-256 `554adb3886491fa686e950773b2d5c5cc759377eae0f3a7d3abf9f15fd353293`.

The driver builds one alternate module changing only the non-flat `Cfg::STAGES` constant, then checks both baseline and candidate against the task reference for batch 1, 16 and 32, query lengths 1 and 8, and seeds 11 and 12. It measures 24 alternating cold-L2 CUDA-graph replays. Flat kernels retain their original depths; any shape using a flat dispatch is a control rather than a stage-two test. This public H200 image with SGLang stub is an exploratory screen; promising results need repeated paired measurement, full official correctness and scoring, and decoder integration checks. The subprocess is capped at 300 seconds; build failures and results are retained in `modal.log`.

Run `ap-glayByJpHLBDLauImSq2qn` completed with return code 0; both variants passed the reference on all 12 cases. Median parent/candidate ratios for seeds 11 and 12 were:

| Batch / query length | Ratios |
| --- | --- |
| 1 / 1 | 0.986 / 0.984 |
| 1 / 8 | 0.984 / 1.006 |
| 16 / 1 | 0.939 / 0.933 |
| 16 / 8 | 0.949 / 0.978 |
| 32 / 1 | 0.966 / 0.977 |
| 32 / 8 | 0.917 / 0.903 |

Timings include this harness's graph/event overhead; ratios compare independently computed medians, not medians of per-round ratios. This screen shows substantial regressions in several cases and no reliable improvement. Retain the scored pipeline and do not submit this variant. It does not rule out a different scheduler or shape-specific pipeline redesign.
