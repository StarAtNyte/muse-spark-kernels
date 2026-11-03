# Single-prepare WGS deferred-flush screen

The earlier two-prepare full-layer screen timed out before producing measurements. This driver prepares the readable decoder once, builds its existing FCI=4 MoE extension and the FCI=5 alternative with the same flags and weight state, and swaps only the batch-32 `b4` extension during graph capture. It times full decoder graphs with alternating read-flushed cold-L2 replays and checks each output against the reference. Candidate source is `../decoder_readable_wgs_fci5_v1/submission.py`.

For batch-32 sliding decode, seed 5101 measured 134.42 → 134.24 µs (1.0013×); seed 5102 measured 143.92 → 144.16 µs (0.9983×). Both versions passed both reference checks. Results are effectively neutral at this precision; reject the FCI=5 variant. The test covers two seeds and one load and does not rule out other schedules or other shapes. Full setup and measurements: `modal.log`.
