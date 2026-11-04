"""decoder_stack_v2 = decoder_stack_v1 (first judged copy, sha 6a2c9ad3) + attn_b1full_v1 'bv' attention hunks
(vs attn_vfull_v1/submission_g16b.py): BF16 v4.bf16x2 split-KV partials for the one-request q_len-1 full-layer builds
(MODE & 11 == 11), and 16-byte v4.bf16x2 reds through a quad transpose for the q_len-8 BF16 partials."""
import pathlib, hashlib, ast, sys
E = pathlib.Path(__file__).resolve().parent.parent
s = (E / "decoder_stack_v1/submission.py").read_text()
s = s.replace("\n# (second judge draw)\n", "\n")
def once(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a)); return s.replace(a, b)
s = once(s, "# decoder_stack_v1 (2026-11-05): =", "# decoder_stack_v2 (2026-11-05): decoder_stack_v1 + attn_b1full_v1 bv attention (BF16 v4.bf16x2 split partials for q_len 8 and the one-request q_len-1 full builds).\n# decoder_stack_v1 (2026-11-05): =")
s = once(s, "  constexpr bool BFP = (L == 8) && !FLAT;   // bf16 split-KV partials (per-request q_len 8 kernels)",
            "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 11);   // bf16 split-KV partials (per-request q_len 8, one-request q_len-1 full)")
s = once(s, """    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc""", """    if constexpr (L == 1) {
      if constexpr (BFP) {   // dims 16 tig + 8 c + 0..7 of this thread: one v4.bf16x2 op each
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          uint32_t u[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) u[e] = bf2u(__floats2bfloat162_rn(o[mt][2 * e][2 * hf + c] * f, o[mt][2 * e + 1][2 * hf + c] * f));
          asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(acc_bf(p, acc + 16 * tig + 8 * c)),
                       "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]) : "memory");
        }
      } else
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc""")
s = once(s, """    } else if constexpr (OKC_RED4) {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;""", """    } else if constexpr (BFP) {
      // bf16 partials: 16-byte v4.bf16x2 atomics of 8 consecutive dims (half the L2 operations of v2.bf16x2):
      // pack each n-tile's dim pair, then transpose the quad's 4x4 blocks of packed pairs (xor 1, xor 2) so lane
      // tig holds n-tiles tig and 4 + tig whole (dims 8t + 0..7)
      uint32_t u[8];
#pragma unroll
      for (int t = 0; t < 8; ++t) u[t] = bf2u(__floats2bfloat162_rn(o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f));
      const bool q1 = (tig & 1) != 0, q2 = (tig & 2) != 0;
#pragma unroll
      for (int h = 0; h < 8; h += 4) {
#pragma unroll
        for (int k = 0; k < 4; k += 2) {
          const uint32_t y = __shfl_xor_sync(0xffffffffu, q1 ? u[h + k] : u[h + k + 1], 1);
          if (q1) u[h + k] = y; else u[h + k + 1] = y;
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint32_t y = __shfl_xor_sync(0xffffffffu, q2 ? u[h + k] : u[h + k + 2], 2);
          if (q2) u[h + k] = y; else u[h + k + 2] = y;
        }
        asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(acc_bf(p, acc + 8 * (h + tig))),
                     "r"(u[h]), "r"(u[h + 1]), "r"(u[h + 2]), "r"(u[h + 3]) : "memory");
      }
    } else if constexpr (OKC_RED4) {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;""")
# fresh extension name so no cached build of the v1 attention source is reused
s = once(s, 'load_inline(name=f"muse_att_v298w1pf_{arch}"', 'load_inline(name=f"muse_att_v298bv_{arch}"')
dst = E / "decoder_stack_v2/submission.py"; dst.write_text(s); ast.parse(s)
print(len(s.encode()), "bytes sha256", hashlib.sha256(s.encode()).hexdigest())
