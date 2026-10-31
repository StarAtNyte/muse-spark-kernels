# Short sliding prefill MoE profile

The candidate check `6a672f2beca74962` attributes about 264 µs of a 582 µs
turn-1 sliding layer to fused MoE. `tools/prioritize.py` calculates that a
hypothetical 20% reduction in the whole case would increase the prefill track
score by 4.75%; this is sensitivity, not available headroom. The previously
archived NCU profile targets 16384 tokens, so it does not establish the
bottleneck for the 753-token shape.

This diagnostic runs the exact published best prefill source (SHA-256
`f49c8edacb851cb979891cb886d43f4329ee82a4c05acc20630e5605a565f7b9`)
on the public challenge image, B=1, context 80000, sliding window, page 1,
query length 753 and seed 9400. After a full reference check, NCU samples
one fused MoE invocation with clock control disabled and records speed-of-light,
scheduler and warp-stall sections. SGLang is stubbed and attention choice is
pinned; the profile is diagnostic, not a judge measurement. The subprocess is
bounded to 275 seconds inside the 300-second runner. Raw output and any errors
are in `modal.log`.

Run `ap-svbnhlEBEW1aqI8aPlmjds` completed with NCU and driver return code 0;
the reference check passed. NCU collected 17 passes of the intended fused MoE
kernel and reported 223.55 µs duration, 71.73% SM compute throughput, 40.77%
DRAM throughput, 2.50 active warps and 0.55 eligible warps per scheduler, and
0.38 issued warp per scheduler per cycle. GPU clocks were not fixed. Its
Warp State section lacked the PC-sampling metric, so it provides no ranked
stall causes. These counters are for one input, profiler replay and the pinned
component setup; they do not prove a bottleneck or predict speedup. NCU's
generic estimated speedup is not evidence that a realizable optimization
exists. A follow-up full-set profile is required to obtain sampled stalls.
