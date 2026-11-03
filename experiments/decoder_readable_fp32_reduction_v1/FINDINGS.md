# Readable decoder with rs128 BF16 reduction disabled

The encoded compact QKV/O source and escaped MoE helper were expanded into readable CUDA, and the rs128 backend that combines BF16 split partials was disabled. This retains the other inherited routes; its full source and precision audit is not complete.

Official private check `bbe34f0cbe494c88` passed 13/13 at 2.23574×, 95% CI 2.20220–2.26928× (10,117.6 µs). The previous historical 2.26032× leaderboard result is not a clean comparison: it used encoded source and the score intervals overlap. The full per-case profiles and logs are archived here. This result establishes correctness on the judge's checks, not compliance of every inherited route, nor a rank gain. No public submission has been made.
