# Third-party source

The `swk` CUDA kernel and its host launcher are adapted from [`lee101/openkernels`](https://github.com/lee101/openkernels), file `muse_attention_decode/submission_v85diag.py`, revision `c565e373d34575c5f7df862b05f8be7619078b0f` (Apache-2.0). Upstream's design and H200 measurements are documented in `docs/attention_fresh_design.md` at that revision. The upstream development file also contains diagnostic sweeps; those are not included in this candidate.

The copied kernel structure is modified here to read K and V scales independently from their device tensors and to route only the two targeted H200 shapes. The full upstream license text is preserved in [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0).
