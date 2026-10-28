# 64-token expert chunks: current designs rejected

All15 independent task checks passed; outputs were bitwise identical to the
32-token baseline in these tests. The control, fragment-streaming and paired-K
versions all lost: normal-routing speedups roughly0.83–0.90x. On the skewed
case they reached0.70/0.77/0.83x respectively. No official check/submission.

The routing evidence matters: at M128, one seed had no expert above32 tokens;
the other had only one. At M256, only two or three exceeded32. Thus the wider
reuse applies to a minority of experts, while changing NTB4 toNTB2 also changes
the many small experts' tiling. These experiments reject the current uniform
dispatch, not every possible hybrid for busy experts. A hybrid would need
preserved small-expert tiling and fresh end-to-end evidence.

The paired-K implementation compiles its targeted NTB2/NMAX8/S3 kernels at
158/160 registers, no stack or spills. Correctness alone is insufficient:
its increased shared-memory work/scheduling changes did not yield a win.
The local driver prunes only unused dispatch instantiations; tested kernels,
weight preparation, routing and finish math remain in the measured pipeline.

## Follow-up hybrid

`hybrid_screen.log` preserves the production 32-token algorithm for all light
experts and uses the paired-K 64-token version only for busy experts. All five
checks passed with identical outputs, but speedups were 0.963/0.970/1.019/0.950
on normal-routing cases and 0.834 on the skewed case. This rejects the tested
hybrid too; no official quota was spent on it.
