# Compact QKV at 256 tokens: rejected

Extended the compact FP32-cluster-reduction QKV route to m=256 using the existing decoder configuration and compared it with the parent `ag` FP32 cuBLASLt pin on H200, cold-L2 alternating graph replays. Both implementations passed all three reference checks. Seeds 61, 62 and 63 measured 22.416 → 25.008 µs (0.8964×), 22.160 → 25.856 µs (0.8571×), and 22.592 → 25.056 µs (0.9017×). Timings include harness overhead. Reject this m=256 route; retain the parent pin and keep the m=128 experiment separate. No public submission was made.
