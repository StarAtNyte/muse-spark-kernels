# Explicit K64 scale hoist

The K64 permutation makes each lane select the same scale pair across the four K16 steps in a K64 stage. The candidate computes that scale pair once before each four-step loop, rather than writing the expression inside the loop. CUDA compilation may already perform this common-subexpression elimination; longer register lifetimes could also hurt occupancy or scheduling. This variant is an experiment, not a promoted kernel.

The paired benchmark loads the candidate twice and reverses only the hoist in the baseline module's CUDA source. Both variants retain the same permutation and preparation, so any difference is attributable to code generation and measurement noise rather than the data layout. The first benchmark is complete. It uses a stubbed attention import and is not the official judge.


H200 first screen (`bench.log`): baseline/candidate ratios 0.9970, 0.9985, 1.0449 and 1.0004 on batch 1 sliding q753, batch 4 full q753, batch 1 full q16384 context 0, and batch 1 full q16384 context 65536, respectively. The apparently large improvement for context 0 needs repetition. One seed and eight paired samples per shape; the attention tuner failed under the import stub and fixed choices were used. These are not official scores and the benchmark did not perform full correctness checks. A three-seed repeat of the context-0 shape is running.

The repeat (`repeat.log`) did not reproduce the 4.49% point estimate: three seeds produced baseline/candidate ratios 0.9987, 1.0017 and 1.0025, with 12 alternating samples per seed. This offers no convincing gain; retain the scored shuffle version and do not submit this variant. The screen did not run the official correctness suite, and a lack of measured gain does not establish whether the compiler hoisted the operation.
