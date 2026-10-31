# Short-prefill mixed-order repeat

Repeat `../prefill_moe_small_order2_v1/` on the highest-weight B=1 sliding
case using seeds 9400, 9407 and 9417. Both variants share preparation and
plans, pass the local reference check, and are compared with the median of 24
alternating cold-L2 replay ratios per seed. The public H200 image and bounded
runner are used. Attention is pinned because the SGLang stub cannot validate
the tuner; this is a full-layer screen, not an official score. Raw output and
any failure are saved in `modal.log`.

Run `ap-uwNsbVXd81RkijXufRNnZX` completed with return code 0; both variants
passed all six reference checks. Median paired parent/candidate ratios for
seeds 9400, 9407 and 9417 were 1.00239, 0.99801 and 1.00397. The small,
mixed results do not establish a reproducible improvement. Retain the scored
order and do not submit the candidate on this evidence. Raw results are in
`modal.log`; this does not rule out other schedules.
