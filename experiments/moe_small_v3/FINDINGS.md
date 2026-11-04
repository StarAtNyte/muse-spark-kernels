# moe_small_v3: new persistent M<=8 MoE kernel (rejected)

Design: 132 CTAs take every 36 KB unit (A units then B units, grouped by expert) in strided order c, c+132, and so on. Warps: producer, epilogue, poller (stages x and inter rows in smem after each expert's A finishes) and credit (one atomic per CTA per expert). Sums of squares via FP32 atomics, then a monotonic grid barrier and a distributed combine. FP32 accumulation and storage.
Equation at bs1 (D=2): 1.0 issue + ~1 first byte + 12 x 1.09 us + 2.9 tail = 18.3 us. Measured: first unit 4.86 us, A done 14.1, B done 19.5, final 21.0. Units take 1.19 (A) / 1.39 (B) us against 1.09 ideal.
Paired vs parent (Modal): 0.84/0.80 first version, then 0.95/0.93 with rows staged in smem, best 0.949 (bs1) / 0.963 (bs8) / 0.955 (bs4). The no-compute ceiling is 1.12 (bs1) / 1.03 (bs8), so this layout cannot win bs8. The strided order doubles expert switches per CTA, and every global load during the stream costs several us.
