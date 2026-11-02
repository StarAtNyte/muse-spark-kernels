# K64 transfer: remaining short-prefill shapes

The main repeat `prefill_keell_k64_confirm_v1` measured the current-best prefill source with our earlier K64 scale-reuse/shuffle-gather change on turn-1 sliding, turn-4 full and first-turn sliding. This complementary H200 screen covers turn-1 full and turn-4 sliding, three seeds each, with two baseline captures, full task-reference checks and 16 randomized cold-L2 graph rounds. Candidate source and transfer details are retained in `prefill_keell_k64_transfer_v1/`.

Both variants share non-MoE weights and plans; the candidate owns repacked expert weights and a compiled extension. SGLang is stubbed and attention is pinned, so the full official judge check remains necessary. The subprocess is capped at 300 seconds; raw results and failures are in `modal.log`.

Run `ap-R1uuXV3BnnOOBvzRgMQGxj` completed with return code 0; all 18 baseline/candidate reference checks passed. Median paired parent/candidate ratios for the three seeds were 1.00562 / 1.00630 / 1.00149 for turn-1 full and 1.00327 / 1.00228 / 1.00292 for turn-4 sliding. A/A ratios were 0.99929 / 1.00006 / 0.99904 and 1.00067 / 0.99825 / 0.99893 respectively. These support small local gains in the two remaining short cases, but the pinned fallback attention and public-image setup mean they do not establish an official score. The full check remains necessary.
