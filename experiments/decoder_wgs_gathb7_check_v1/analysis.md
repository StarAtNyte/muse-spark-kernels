# Decoder WGS gather batch 7

Transferred only `OKC_GATHB=8` to `OKC_GATHB=7` in the published decoder.
The isolated full H200 check `04a138c2ac314122` passed correctness and all 12
timed cases at **2.08635x** (95% CI **[2.05506, 2.11765]**). The prior
same-source check was **2.09091x** (95% CI **[2.05955, 2.12228]**). The new
point estimate is lower and the intervals overlap, so this cross-kernel
transfer does not establish a gain. Keep the published gather batch and do not
submit this candidate.

The profiler still shows the MoE kernel as a major part of the integrated
decoder at several shapes (for example, 207 us / 40% of reported GPU time on
verify-32 full). That supports continued WGS dataflow work, but this one-value
gather change is not enough. Exact candidate, judge status, and source hash are
preserved here.
