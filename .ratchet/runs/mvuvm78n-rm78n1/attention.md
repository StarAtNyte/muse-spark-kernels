# Attention worker evidence log

Current source is a fused CUDA implementation, not the older FA3 hybrid described in the assignment. Supported shapes already fuse QK normalization, RoPE, new KV, attention and output. Parent is scored_attention_plan16_v1, public 2.6176787949x.

Read TASK/AGENTS, reference math/checker, current source, attention math/priorities/screen, PTX campaign, H200 lower bounds and current-attention snapshot. Inspected archived status outcomes (findings/w2-work/prior-statuses.json), plus latest full-verify NCU, page shuffle, prefetch and full-decode grid failures. Older reports refer to superseded implementations.

Recomputed scored sensitivities in findings/w2-work/priorities.json. Highest single-case sensitivity: full decode32 (20% less time => +3.16% score), full verify32 (+3.00%), full decode16 (+2.89%), full verify16 (+2.77%). Small sliding has greatest ratio to the theoretical floor, but this floor excludes practical gather/launch costs.

Revision attention_page_lookahead2_w2_v1: increase producer page-index lookahead from one to two tiles on the flat verify path. Preserves all math and addresses. Modal screen started, batches 3/16/32/33, page1 coherent full contexts, cold-L2 paired graph replays, reference checks. Exact source hashes in findings/w2-work/provenance.json.

## Modal result

App ap-p6nTDpb2ecsSiseYdlUAdd completed. All four local reference checks passed. Paired cold-L2 graph baseline/candidate ratios: batch3 0.992, batch16 1.002, batch32 1.001, batch33 0.998. These are one-seed screens with graph overhead included, not official scored speedups. No speedup established.

Official private check launched through tools/experiment.py as 75a9e2131a594077. Full validation was explicitly requested; no public submission is authorized by these neutral performance results. Source remains immutable.

Live authenticated board snapshot: StarAtNyte rank99, 2.6177x; leader 3.2221x. Stored findings/w2-work/leaderboard.json.

## Final official outcome

Official check 75a9e2131a594077 passed 16 tests and 12 timed cases; speedup 2.6018471380x, CI [2.5628194310, 2.6573917030], versus published 2.6176787949x. This cross-run comparison alone is not a significant regression claim; the paired Modal screen is also neutral. Reject for promotion. No public submission; no production source changes.
