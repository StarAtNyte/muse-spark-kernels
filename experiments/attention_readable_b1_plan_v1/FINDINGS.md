# Readable attention: batch-one sliding launch plans

Using the readable attention parent, the judge image and H200, compared nine launch plans each for q_len 1 and 8 at 80K context / 2,048-token sliding window. Every retained plan passed the reference. Cold-L2 graph timing used 12 alternating replays and one seed. The baseline was 8.83 µs (decode) and 10.54 µs (verify), including harness timing overhead; the next best was 8.93 and 10.61 µs respectively. This screen found no improvement; retain the parent. The one-seed results do not rule out a smaller gain or other launch plans.

Driver and full Modal log are in this directory. The independent official parent check passed 16/16 at 2.60414×. No public submission was made.
