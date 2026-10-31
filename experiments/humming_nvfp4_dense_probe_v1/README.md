# Humming Hopper BF16 × NVFP4 feasibility probe

[Humming](https://github.com/vllm-project/humming), Apache-2.0, commit
`948e6fd547e703065be51529ac1db5569256774e`, supports BF16 activations,
E2M1 weights and E4M3 group scales in its public schema; its SM90 heuristic
tests include the group-16 configuration. The [Humming release article](https://vllm-project.github.io/2026/10/19/humming.html)
reports strong H200 W4A16 results on other workloads. Those numbers do not
establish performance on Muse's NVFP4 experts. Its W4A8 path is not eligible
for this task: `kernels/muse_prefill_layer/TASK.md` requires BF16 activations
outside the allowed FP8 attention queries and cache.

This isolated research probe loads exact randomly generated Muse-format FP4
codes, E4M3 block scales and FP32 global scale into Humming, uses BF16
activations, and checks its output against a dequantized BF16 `torch.mm`
comparator, including a one-hot row. It measures 4096×4096 dense GEMM at
M=8/32/128/256 on the public H200 image, using alternating cold-L2 graph
replays. The materialized BF16 weight is a diagnostic comparator only; it is
not a valid submitted weight representation. This is not the routed, fused
MoE workload and cannot predict a challenge score. The one-hot check uses
the materialized BF16 comparator and does not replace official correctness.

The runner requires a local checkout at `/tmp/muse-humming-research` and
verifies the pinned commit before mounting its source. It does not assume
Humming is installed on the judge, and no submission imports it. Any useful
technique would need a compliant self-contained implementation, license
attribution, end-to-end validation, and official checks. The GPU subprocess
is bounded to 300 seconds; raw output and any failure are in `modal.log`.

The first launch (`initial_failure.log`) did not reach the benchmark: module
import on the remote worker tried to run the local checkout's Git hash check
against a path absent there. The run was interrupted to prevent retries. The
runner now performs that verification in the local entry point before the
remote function is invoked; the corrected run is pending.
