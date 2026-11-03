# Earlier WGS deferred-flush schedule

Compared main MoE `OKC_WGSFCI=3` against the readable decoder's 4 with one shared preparation, alternate extension capture, cold-L2 graph replays and reference checks. Batch-32 sliding decode measured 134.93 → 134.88 µs (1.0004×, seed 5101) and 144.56 → 144.11 µs (1.0031×, seed 5102); both outputs passed. The ~0.17% geometric gain is too small and variable to justify promotion or a public run. These two seeds and this one load do not rule out different behavior elsewhere. Full driver and log are retained.
