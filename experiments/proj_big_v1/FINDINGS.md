# proj_big_v1: 12-bit wgmma GEMM for m=128/256 projections (rejected)

Floors: m=128 BF16 -> 12-bit byte floor qkv 8.4->6.5, o 4.0->3.1, pooler/unpooler 14.6->11.1 us. m=256 is compute-bound (9.8/4.3/17.4/17.4 us), so 12-bit gives nothing there. Decode is 1.5 int ops per weight. The binding limit is L2 re-reads of x (pooler m=128: 67 MB of x through L2 against 50 MB of weights).
Design: 2 consumer WGs (m64n128k16, register-A), a producer warp (TMA x + bulk 12-bit weights), an escape warp; vjeux decode8 into A fragments; split-K with red.global.add.v4.bf16x2.
Paired vs parent (>1 faster): m128 qkv 0.84, o 0.64, pooler 0.96, unpooler 0.97; m256 0.75/0.66/0.73/0.76.
Why: a fixed ~270 ns per 64-k pipeline stage even with no loads or wgmma (source unknown), a 5-8 us epilogue (escapes via smem atomics), clusters running in 2 waves (120 CTAs resident), multicast 1.5-1.8x slower, and 4-WG tiles exceeding the register cap.
