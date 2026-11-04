# Scored standalone MoE with FP32 per-expert storage

Source SHA-256 `2c6ebcb747a4e436e84fb1ee80d8dcb6727fde538a62d0fc73803f5b88432b96` matches the checked source byte-for-byte. Private check `ce7e9f4df22a4ee3` passed 5/5 at 2.38300× (95% CI 2.34726–2.41875×). The candidate uses FP32 per-expert WGS storage before normalization and combination, retaining the existing FP32 fixup path. The earlier targeted H200 screen showed no meaningful speed difference from the BF16-storage parent; that does not prove the earlier format was prohibited.

Scored job `5c2e8be4c4994a50` submitted and pending. Passing tests and code inspection do not guarantee final organizer eligibility or all-input correctness.
