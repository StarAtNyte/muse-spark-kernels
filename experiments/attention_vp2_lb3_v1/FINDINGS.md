# Attention VP2 three-block launch bound

Candidate SHA-256 `58b1caebb874188942936c321891b9ae58c07a16dac46661728e8b09bd728e65` changes VP2 launch bounds from `(256,2)` to `(256,3)` relative to `attention_vp2_lb2_v1`. Paired public-image H200 measurements on three batch-1 sliding verify inputs passed reference checks and gave two-block/three-block ratios 1.008, 1.020, and 1.002. Actual occupancy and spills have not been measured. Official private check `f350069f2f614336` is pending; the two-block candidate's scored run `6f2b2edf215d4ea3` is also in flight. No new public score is claimed.

Driver, source and raw output are in this directory.
