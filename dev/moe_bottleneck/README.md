# Large MoE controlled ablations

Diagnostic only: all ablated outputs are incorrect and must never be submitted.
Frozen parent: experiments/moe_wm_v22/submission.py. H200, cold-L2 graphs, 24 alternating paired replays; seeds 419 and 7919, M=128 and 256. Four full unchanged-pipeline checks passed (relative L2 ~0.0055).

| M | Stage | Removed | Median latency reduction |
|---|---|---|---:|
| 128 | gate/up | FP4 conversion (replaced with synthetic operands) | 7.1% |
| 128 | gate/up | MMA (replaced with XOR sink) | 3.1% |
| 128 | gate/up | Conversion + MMA | 11.4% |
| 128 | gate/up | Scale loads + extraction (constant scales) | 10.6% |
| 128 | down | FP4 conversion (replaced with synthetic operands) | -0.9% |
| 128 | down | MMA (replaced with XOR sink) | 2.3% |
| 128 | down | Conversion + MMA | 6.7% |
| 128 | down | Scale loads + extraction (constant scales) | 7.2% |
| 256 | gate/up | FP4 conversion (replaced with synthetic operands) | 5.5% |
| 256 | gate/up | MMA (replaced with XOR sink) | 7.3% |
| 256 | gate/up | Conversion + MMA | 13.9% |
| 256 | gate/up | Scale loads + extraction (constant scales) | 9.2% |
| 256 | down | FP4 conversion (replaced with synthetic operands) | 18.9% |
| 256 | down | MMA (replaced with XOR sink) | 14.7% |
| 256 | down | Conversion + MMA | 26.0% |
| 256 | down | Scale loads + extraction (constant scales) | 12.3% |

Interpretation: gate/up retains most latency even without conversion and MMA. Investigate weight/scale staging, bandwidth and occupancy before redesigning its arithmetic. At M256 down projection is more conversion-sensitive. Constant scales save roughly 5–11%, so scale transport and extraction deserve a targeted experiment.

These are not additive cost shares or rigorous upper bounds: ablations change register pressure (baseline down 157 registers vs synthetic-conversion 158, no-MMA 88, no-both 82, constant-scale 142), dependency chains and instruction scheduling. Isolated stage inputs retain the real baseline intermediate. Cold-L2 stage times do not equal in-pipeline stage times. No official quota spent; one GPU job completed.

Evidence: screen.log; compile.log; driver ../modal/drv_moe_bottleneck.py.
