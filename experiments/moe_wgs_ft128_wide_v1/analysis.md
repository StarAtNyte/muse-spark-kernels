# Complete 128-token WGS dispatch screen

This candidate adds `WGSMMA<9>` through `WGSMMA<16>`, dispatch for 9–16 eight-row tiles, and a bounded cost-table lookup to the two-weight/two-activation-slot 128-token variant. It was compiled on CPU and transferred to H200. All six normal paired cases passed the task reference with bitwise-identical baseline/candidate outputs. A further adversarial input forces 128 rows to local expert 0; both implementations passed the reference, and candidate/baseline relative L2 difference was 0.0004897. This test covers the previously missing 128-row dispatch but does not replace the full official suite.

For seeds 419 and 7919, median paired baseline/candidate ratios were 0.9634 and 0.9660 at M32, 0.9325 and 0.9241 at M128, and 0.9166 and 0.9020 at M256. The candidate is approximately 4–11% slower than the scored baseline on these normal cases. Do not promote or submit; retain the baseline. The larger tile has higher shared-memory footprint and wider register allocation, but this screen does not isolate the cause of the slowdown.

Candidate SHA-256 `0ba49c73f18af5e9659289191e5eba3174bdd918d880b69667a72a06cdd02ebd`; baseline SHA-256 `a7fa7eac2b2f6a943ceb977197d992d2fe770aea7a1f65a82ae9d4462127d48a`. Modal app `ap-k376CQzrkXbnkjV0vkP7O2`; full output `screen.log`; exact runner `runner.py`.
