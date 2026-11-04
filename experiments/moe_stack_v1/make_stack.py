"""Builds moe_stack_v1/submission.py from scored_moe_fp32_ybuf_v1: (a) DF=1 in TINY2 (M 5..8, incl. the bs8 PS build),
(b) WGSFINPAR (phase-B epilogue sum-of-squares atomics + column-parallel final) for M 17..32 only, with its own
two-parity ss region (ssF) keyed by the MFIN launch parity and zeroed by every MFIN launch."""
import sys
src = open("experiments/scored_moe_fp32_ybuf_v1/submission.py").read()
a0 = src.index("CUDA_SRC = r'''"); a1 = src.index("\n'''", a0)
head, cu, tail = src[:a0], src[a0:a1], src[a1:]
def rep(s, old, new, n=1):
    c = s.count(old); assert c == n, (c, old[:80]); return s.replace(old, new)
# knob
cu = rep(cu, "#ifndef OKC_MFIN\n", "#ifndef OKC_WGSFINPAR\n#define OKC_WGSFINPAR 0  // WGS: phase-B epilogue sums of squares (atomics into ssF) + column-parallel final (M <= 32 builds)\n#endif\n#ifndef OKC_MFIN\n")
# Params field
cu = rep(cu, "  float* ss1;      // [EMAX]\n", "  float* ss1;      // [EMAX]\n  float* ssF;      // [2 EMAX] WGSFINPAR sums of squares, one copy per MFIN launch parity\n")
# this launch's ss copy in FINPAR builds
cu = rep(cu, "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n", "    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n#if OKC_WGS && OKC_WGSFINPAR\n    S.ssp = p.ssF + (gen & 1) * EMAX;\n#endif\n")
# zero the other parity's ssF in every MFIN launch
cu = rep(cu, """      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
""", """      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) p.ssF[o * EMAX + i] = 0.f;  // WGSFINPAR: next launch's ss
""")
# epilogue atomics
cu = rep(cu, """        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
#endif
      }
    }
""", """        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
#endif
      }
#if OKC_WGSFINPAR
      {  // this tile's 16 rows of the route's sum of squares: reduce over the 8 lanes holding them (g), one atomic
        const float y0 = d4[h] * w0, y1 = d4[2 + h] * w1;
        float sq = tk < cnt ? y0 * y0 + y1 * y1 : 0.f;
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (g == 0 && tk < cnt) atomicAdd(ss_cur() + jst + tk, sq);
      }
#endif
    }
""")
# final pass
old_fin = "#if OKC_WGS && OKC_WGSFINSS\n  // item = one token's whole row"
new_fin = """#if OKC_WGS && OKC_WGSFINSS && OKC_WGSFINPAR
  // item = (token, 1024-column segment), thread = 2 columns: the sums of squares came from the phase-B epilogues
  {
    constexpr int CPT = 2, SEGC = NT * CPT, FSEGN = HD / SEGC;
    static_assert(HD % SEGC == 0, "segments tile the row");
    for (int it = blockIdx.x; it < p.M * FSEGN; it += gridDim.x) {
      const int tk = it / FSEGN, col = (it - tk * FSEGN) * SEGC + tid * CPT;
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {  // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
      const float* ssv = ss_cur();
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(__ldcg(ssv + q) * (1.f / HD) + 1e-8f);
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));
          a0 += r * v.x; a1 += r * v.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(a0, a1);
    }
  }
#elif OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row"""
cu = rep(cu, old_fin, new_fin)
# host: ssF after the MFIN flag region
cu = rep(cu, "  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), \"MFIN needs the flag region in scratch[0]\");\n",
         "  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS + 2 * okc::EMAX), \"MFIN needs the flag region + ssF in scratch[0]\");\n  p.ssF = reinterpret_cast<float*>(ib + 8 + 1024 + okc::EMAX + okc::MF_INTS);\n")
src = head + cu + tail
src = rep(src, "    return [torch.zeros(8 + 1024 + EMAX + 4096, dtype=torch.int32, device=dev),",
          "    return [torch.zeros(8 + 1024 + EMAX + 4096 + 2 * EMAX, dtype=torch.int32, device=dev),  # + WGSFINPAR ssF (2 EMAX)")
src = rep(src, 'TINY2_DEFINES += ("-DOKC_DF=0", "@var")', 'TINY2_DEFINES += ("-DOKC_DF=1", "@var")  # moe_stack_v1: fence-free A->B hand-off (M 5..8)')
src = rep(src, "PS_SHAPES = {8: ('tiny2', {}, 't2_wsi')}",
          "PS_SHAPES = {8: ('tiny2', {}, 't2_wsi')}\n# moe_stack_v1: M 17..32 run the WGS build with the parallel final pass (WGSFINPAR); larger M keep the base final\nPS_SHAPES.update({_m: ('main', {'OKC_WGSFINPAR': 1}, None) for _m in range(17, 33)})")
open("experiments/moe_stack_v1/submission.py", "w").write(src)
print("ok")
