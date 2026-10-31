# Forge prefill run final state

The resumed `muse_prefill_boundary_261030-211439-b293` run stopped on its configured budget with no accepted candidate or harvested kernel. The first student attempt failed compilation; after a launch-only smoke test, the second candidate failed numerical correctness at its focused gate. `status.jsonl`, `report.md`, `run.json`, and `run.log` preserve the final run state. Detailed candidate artifacts and job logs are in `experiments/forge_prefill_iter0_v1/`, `experiments/forge_prefill_iter1_v1/`, `experiments/forge_prefill_iter2_probe_v1/`, `experiments/forge_prefill_iter2_smoke_pass_v1/`, and `experiments/forge_prefill_iter2_gate_v1/`.

The recorded best prefill submission remains unchanged. A separate stream-repair diagnostic is underway; it is not part of this Forge run and has not established correctness or a speedup.
