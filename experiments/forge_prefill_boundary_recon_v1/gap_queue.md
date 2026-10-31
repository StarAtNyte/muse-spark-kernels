# Out-of-run gap queue (recon-table rows this run did not verify: muse_prefill_boundary)

Rows with VERIFY `none` have no machine-read acceptance; rows that say "may not be expressible" without an `[EXPR:]` ladder verdict are unverified guesses; rows with VERIFY `inferred` are structural readings whose discriminating observation gave no positive reading (the two readings are byte-identical). None of the three may be used as a conclusion. These rows are the recon's reading of the reference's structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified recipe carries knowledge across runs; to verify one, start a separate probe, with the evidence under `docs/kb-provenance/`.

- **Estimated usage** = 54–60 regs/thread — VERIFY `none`; notes: Source-derived working-set accounting (b[16] + s[16] + temporaries ~54–60); actual NCU `sm__registers_per_warp` not captured (log truncated) (no machine-read verification in this run)
- **Non-accumulator working set** = ~54–60 (all registers; no MMA accumulator) — VERIFY `none`; notes: LIVE-WAIVED; no MMA chains (no machine-read verification in this run)
