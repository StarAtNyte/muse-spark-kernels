# Decoder compact T128 projections + tiny warp order

Candidate SHA-256 `febc0d6edc62ce08a698f8868747cff412bbc24a5c834dbf02c1ca88a581fbde` stacks the compact T=128 QKV and output projection routes from `decoder_stack_compact_qkvo_v1` with the M<=8 tiny MoE warp-order change from `decoder_tiny_original_warp_v1`. The routes affect different token counts, but preparation and integration must still be checked together.

Official private check `c411f676f4374f35` passed 13/13 at **2.26054×**, 95% CI **[2.22664, 2.29445]**, versus public best **2.26032×**. The near-equal point estimates and overlapping intervals do not establish a gain. The separately checked compact candidate scored 2.271× in its check and showed a 1.00294× selected-case paired H200 gain; the tiny candidate checked at 2.25398× and showed selected small-batch gains in a separate paired screen. These separate measurements do not prove the stack's combined gain.

Scored quota was exhausted when checked. Preserve the exact candidate and full profiles for a scored submission when quota returns; no public leaderboard result has been claimed.
