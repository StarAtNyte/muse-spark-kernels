# FP8 probability / joint layout outcome

Official private check `cf508e81d0ae4849` passed full correctness and timing
coverage. Aggregate speedup 1.7112269500x, CI [1.6855585, 1.7442675], versus
production private 1.7096881132x. The 0.09% aggregate difference does not
establish a meaningful improvement; no public submission or promotion made.
The local 3–4% large-verify gain did not translate to a comparable aggregate
gain. Keep the precision audit and validated operand mapping as reusable work.

The V-only 64-byte swizzle screen passed task checks but did not consistently
improve on the serial joint layout: batch16 170.72 vs171.30 us, batch32
323.76 vs323.22 us, one seed. No further GPU confirmation warranted.

Next architectural experiment: `../cluster_attention`, fusing partial attention
and merge using DSM. Its separate merge probe passed, but standalone DSM was
slower; only the integrated end-to-end test can establish usefulness.
