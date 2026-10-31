# Forge prefill smoke probe

Forge job `1793427817` tested workspace commit `dc40866` using `notes/smoke_m0gate.py`, after the untested candidate revision in `experiments/forge_prefill_iter1_v1/`. The job failed in the first `torch.randn` call with `CUDA error: no kernel image is available for execution on the device`; it did not reach the candidate invocation or the `PROBE PASS/FAIL` line. The log identifies an H200, CUDA 13.0 and PyTorch 2.14.1+cu130. The cause is not yet established. The test checks launch and output shapes only, not numerical correctness or speed.

This is a failed smoke test without a candidate verdict. Full correctness and timing remain necessary before promotion. The local job log also records waiting for the shared GPU lock; it subsequently acquired the lock and launched the Modal worker.
