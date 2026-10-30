# KPERM3 with activation gather batch 7

The public alygg77 KPERM3 candidate retains GATHB=8. Our previous non-permuted WGS source benefited from GATHB=7, but KPERM3 changes activation gather addresses and therefore the previous result cannot be transferred by assumption. This bounded screen builds both variants from the exact public source with identical packed weights, verifies both against the task reference and times alternating cold-L2 graph replays at M32/128/256. One seed is exploratory; passing full official checks and additional seeds are required before promotion. Upstream source and Apache notices are preserved in `experiments/moe_alygg_kperm3_v1/`.

Source SHA-256 `b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`, driver SHA-256 `c42ddc1d1cb124e2fde4e827bef4ace3b560279e8c0c2dc5e074343e11522f1f`; public-image app `ap-Qrlaos1YROuae0cUR95UPX`. Its output is `screen.log`.
