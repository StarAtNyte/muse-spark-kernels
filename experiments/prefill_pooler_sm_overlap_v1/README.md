# Prefill pooler SM target and router overlap

The scored prefill implementation launches router GEMM and top-k on a side stream alongside pooler GEMM. A configurable cuBLASLt SM-count target exists but is disabled. This experiment tests target counts 116, 100 and 84 SMs (nominal reservations of 16, 32 and 48 on the 132-SM H200), asking whether improved router/pooler overlap shortens the actual layer critical path. The target guides cuBLASLt's plan selection and is not assumed to enforce an exclusive SM partition; its first heuristic may also change the GEMM implementation.

Starting from the exact scored prefill source, the public-image driver shares all prepared state except the pooler plan. It tests turn-1 sliding and turn-4 full, two seeds each; all outputs are checked against the task reference. Two independently captured baseline graphs serve as an A/A control, and 12 cold-L2 graph rounds randomize order among the baselines and candidates. The SGLang stub and pinned attention limit equivalence to the official judge; full correctness and official scoring are necessary for any promotion. The subprocess timeout is 300 seconds; see `modal.log` for plans, samples and failures.

Run `ap-bOAtnftaWUVAkfXhBqDgKg` completed with return code 0; all 20 reference checks passed. Median paired baseline1/variant ratios for the two seeds were:

| Tokens | A/A baseline | Reserve 16 | Reserve 32 | Reserve 48 |
| ---: | --- | --- | --- | --- |
| 753 | 1.0013 / 1.0006 | 1.0032 / 0.9977 | 0.9905 / 0.9895 | 0.9676 / 0.9649 |
| 3012 | 0.9993 / 0.9991 | 0.9936 / 0.9931 | 0.9798 / 0.9816 | 0.9672 / 0.9688 |

The smallest target adjustment is mixed or slower; larger changes regress consistently. No candidate is promoted or submitted. This screen does not distinguish the effects of a changed heuristic from actual concurrency or prove that no other scheduling scheme could help.
