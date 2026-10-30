# Explicit stage-scale reuse in compressed MoE WGS

The candidate computes the four packed BF16 scale pairs once per weight stage and reuses them across its four WGMMA groups. The paired driver reverses only this hoist in the comparison CUDA source, so both paths use the same compact weights and prepared state. H200 app `ap-2WmsTury2YaGbc2UJq3oVC`.

Both paths passed the reference check and produced bit-identical outputs for all six tested cases (one weight seed, two input/routing seeds). Twenty alternating cold-L2 graph replays per case produced baseline/candidate ratios 1.0011 and 1.0015 at 32 tokens; 1.0054 and 0.9964 at 128; 0.9995 and 1.0020 at 256. The small mixed differences do not establish a gain or prove whether the compiler already hoisted the operations. No official check was launched; the promoted MoE remains unchanged. Exact source, driver and output are retained here.
