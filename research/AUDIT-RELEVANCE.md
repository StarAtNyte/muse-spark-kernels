# User-linked OKC audit: relevance to this campaign

Inspected https://github.com/msaroufim/openkernelchallenge-audit at commit
0a68efe41f9372478e3986283c954f39df7484a7. Read source only; no diagnostic
submissions or bundled CLI binary executed.

The published untrusted worker SHA256 is `950437610f57ddffbc1176b65d9c2ae909c47bf67b1e3fe0716a81946986f568`, exactly matching the
worker manifest of our official decoder_pre_norm_v1 job. This is useful evidence
for matching local setup: TF32 disabled, two eager warmups, CUDA-graph capture
on a separate stream, an untimed upload replay, and retained output buffers.
The parent evaluator and its timing/check policy are not included. The audit's
README does not establish shared private implementation with KernelBot; the
available-source token comparisons report no matches at the tested thresholds.

The repo contains protocol/provenance probes and evaluator analysis, not faster
Muse kernels. Its seed formula and failure-code observations do not reduce GPU
work. Do not spend optimization quota reproducing provenance probes. Use the
public worker to improve fidelity of local measurements; use GPU MODE's actual
kernel code and architectural references for algorithm ideas.

This audit does not explain our performance deficit or invalidate competitors'
scores. Our next performance experiment still needs an end-to-end headroom
estimate and a discriminating bottleneck measurement.
