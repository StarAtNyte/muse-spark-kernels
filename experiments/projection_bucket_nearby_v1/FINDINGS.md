# General projection bucket routes: nearby-shape screen

Compared the V7 bucket dispatcher with the same prepared V7 state's ordinary chenwainuo fallback, on H200 with cold-L2 paired CUDA-graph replays. Five layers and token counts 7, 9, 15, 17, 31, 33, 63, 127, 129, 200, 255 and 300 were tested: all 60 candidate and 60 fallback outputs passed reference checks. The 23 changed routes gaining more than 7% showed 1.17–1.58× on router and 1.23–1.46× on dense cases; complete per-case results are in `modal.log`. One seed per shape, with event-timing overhead; results do not prove performance on all nearby sizes or final submission eligibility.

The bucket rules replace the earlier 973 individual off-score selections with a few token-range conditions using prepared backends. They retain the scored QKV-32 and QKV-128 implementations. Full private check `a6d1ceab0bc24fb5` is pending. Remaining source and precision paths require review before public submission.
