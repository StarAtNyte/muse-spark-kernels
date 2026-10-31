# KPERM3 WGS stage-scale hoist

The scored MoE source is the parent (SHA-256
`ac008155520981cd67928bd52bb647efa64096add5acc8edabc779261105792f`).
Each WGS stage loads four packed scale pairs that are reused by four code
groups. The candidate explicitly forms those four BF16 scale pairs before the
group loop, then indexes them by step, instead of expressing the construction
inside the loop. This may reduce repeated scale decode instructions, but the
compiler might already hoist them and extra live registers could cost more.

`driver.py` changes only the alternate WGS source, uses the same prepared
weights, retains the normal shape dispatch, checks each output against the
reference and times alternating cold-L2 CUDA graphs at 32/128/256 tokens and
two seeds. A positive screen would require full official validation; output is
saved in `modal.log`. The subprocess has a 300-second limit.

H200 run `ap-ysnZkY8Re2U71DcGoqs8Fn` completed with return code 0.
Both variants passed the reference in all six cases and had zero observed
relative output difference. Median paired parent/candidate ratios:

| Tokens | Seed 419 | Seed 7919 |
| --- | ---: | ---: |
| 32 | 1.0107 | 1.0008 |
| 128 | 1.0027 | 1.0001 |
| 256 | 0.9989 | 1.0007 |

These differences are too small and inconsistent to establish a useful gain;
the parent stays promoted and no official run was launched. Ratios are the
median of paired ratios, not the ratio of the separately displayed medians.
The screen does not show whether the compiler already hoisted the original
instructions; that would require instruction-level comparison.
