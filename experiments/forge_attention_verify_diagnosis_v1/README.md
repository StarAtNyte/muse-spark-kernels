# Forge attention verify calibration diagnosis

This H200 run compares the exact scored attention source on the same three `verify1` groups and seeds used by Forge calibration. `driver.py` records the challenge's `check_implementation` verdict, relative-L2 and normalized maximum output errors, and the mathematical judge's distance from the challenge reference.

All three cases passed the challenge's local checker. For typical / short / scaled inputs, optimized-versus-judge relative L2 was 0.0444 / 0.0524 / 0.0472, while the normalized maximum error was 0.1006 / 0.1199 / 0.1236. The checker bounds were 0.123 / 0.142 / 0.133, respectively, and the independent judge's relative L2 from the BF16 challenge reference was 0.00090 / 0.00020 / 0.00057. The local checker also checked new cache entries. Full evidence is `modal.log`; this does not establish coverage beyond these three cases or constitute an official submission.

These results show that Forge's generic maximum-error calibration rejects this incumbent on cases accepted by the challenge checker. They do not justify loosening the generic threshold: the challenge uses a dynamic FP8-query-aware L2 bound and separately checks new cache entries. A usable Forge adapter needs that actual correctness contract, independent copies for mutated caches, and validation against the complete official suite before promotion.
