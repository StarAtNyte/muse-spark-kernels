# Baseline and decoder integration audit — 2026-10-31

Use `python tools/baselines.py --forge /home/nitiz/Projects/Self/Challenges/ForgeKernel`
before an experiment. The command verifies source hashes against `best.json`;
its printed path is the recorded best source. The working projection file is
different, so do not assume every `kernels/*/submission.py` is the best source.

The prepared Forge projection, attention and prefill references matched their
recorded best hashes at this audit. Forge's attention task covers only a subset
of the official shapes; passing it is a screen, not a full challenge check.
MoE and the whole decoder each have a recorded best source, but no prepared
Forge reference was verified for them by this command.

The scored decoder baseline (`scored_decoder_compact_qkv_o_v1`, 2.26032x)
already includes KPERM3 and compact QKV and output projection routes. Its
attention and MoE are embedded in the decoder source: updating the standalone
submissions does not update the decoder automatically.

The transfer experiment `decoder_w4_gath7_plan16_v1` stacked the standalone
GATHB=7 MoE gather change and the 16-request sliding decode attention plan on
that scored decoder. Its 24-case paired H200 screen passed, with a small gain
on sliding decode 32 and other cases approximately neutral. Official check
`56bffe6cfc6a4eae` scored 2.19787x, CI 2.16491–2.23084, below the incumbent's
separately scored 2.26032x, CI 2.22642–2.29423. The candidate was not promoted.
These are separate runs with overlapping confidence intervals, not a paired
measurement of the difference: the result does not establish that the changes
are intrinsically slower. Revisit them only with a stronger paired end-to-end
signal or in combination with a substantive new change, and run the complete
official check before promotion.

Evidence: `experiments/decoder_w4_gath7_plan16_v1/manifest.json`, its preserved
official profiles, and `experiments/decoder_w4_gath7_plan16_screen_v1/`.
