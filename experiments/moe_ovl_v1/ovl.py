
# ---- moe_ovl_v1 patches on the WGS MAIN consumer (wgs_seg stage lambda)
# OKC_DAN: dual accumulators for NTT <= OKC_DAN (commit group ci accumulates into d (ci even) or d2 (ci odd); the two
#          groups in flight are independent, so the tensor pipe is not serialized on one accumulator chain)
# OKC_ERN: early weight-slot release for NTT <= OKC_ERN (all 4 code groups loaded at stage start, slot freed at once)
def patch_ovl(a):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_DAN\n#define OKC_DAN 0\n#endif\n#ifndef OKC_ERN\n#define OKC_ERN 0\n#endif\n")
    a = rep(a, "  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma",
            "  constexpr bool DA = NTT <= OKC_DAN, ER = NTT <= OKC_ERN;\n  float d2[WGS_NT][4];\n  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma")
    old = "    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    // KPERM3"
    new = ("    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    uint4 cwe1, cwe2, cwe3;\n"
           "    if constexpr (ER) {\n      cwe1 = *reinterpret_cast<const uint4*>(pc + ((4 + wp) * 32 + lane) * 16);\n"
           "      cwe2 = *reinterpret_cast<const uint4*>(pc + ((8 + wp) * 32 + lane) * 16);\n"
           "      cwe3 = *reinterpret_cast<const uint4*>(pc + ((12 + wp) * 32 + lane) * 16);\n    }\n    // KPERM3")
    a = rep(a, old, new)
    old = ("    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);\n#endif\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n"
           "      const uint4 cwg = cwn;\n")
    new = ("    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);\n#endif\n"
           "    if constexpr (ER) { __syncwarp(); if (lane == 0) mbar_arrive1(S.bars + BW_E + ws); }\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n"
           "      const uint4 cwg = (!ER || lg == 0) ? cwn : lg == 1 ? cwe1 : lg == 2 ? cwe2 : cwe3;\n")
    a = rep(a, old, new)
    old = ("      if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n      } else if (lane == 0) {\n")
    new = ("      if (ER) {\n      } else if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n      } else if (lane == 0) {\n")
    a = rep(a, old, new)
    old = ("          if (FIRST && ci == 0 && jj == 0)\n            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\n          else\n"
           "            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n"
           "                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n")
    new = ("          if (DA && (ci & 1)) {\n            if (FIRST && ci == 1 && jj == 0)\n"
           "              WGSMMA<NTT>::mma0(d2, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3], xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n"
           "            else\n              WGSMMA<NTT>::mma(d2, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3], xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n"
           "          } else\n" + old)
    a = rep(a, old, new)
    old = "  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8)"
    new = ("  wg_wait<0>();\n  if constexpr (DA) {\n#pragma unroll\n    for (int nt = 0; nt < NTT; ++nt) { d[nt][0] += d2[nt][0]; d[nt][1] += d2[nt][1]; d[nt][2] += d2[nt][2]; d[nt][3] += d2[nt][3]; }\n  }\n"
           "  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8)")
    a = rep(a, old, new)
    return a

# OKC_FIN1 (needs OKC_YBF): the phase-B epilogue writes each route's sum of squares as 256 plain per-warp partials
# (rows of a 16-row group, shuffled over the 8 row lanes; no atomics) into the free upper half of ybuf; the final pass
# then reads every route's 256 partials and its BF16 y row together: one L2 round trip and one barrier per token
def patch_fin1(a):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_FIN1\n#define OKC_FIN1 0\n#endif\n")
    old = ("        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n#endif\n      }\n    }\n")
    new = old + ("#if OKC_FIN1 & 1\n    {\n      float* ssb = p.ybuf + (size_t)EMAX * HD / 2;\n      float v[2];\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n"
                 "        v[h] = 0.f;\n        if (tok0 + h < cnt) {\n          const float a0 = __bfloat162float(__float2bfloat16(d4[h] * w0)), a1 = __bfloat162float(__float2bfloat16(d4[2 + h] * w1));\n"
                 "          v[h] = a0 * a0 + a1 * a1;\n        }\n#pragma unroll\n        for (int o = 4; o < 32; o <<= 1) v[h] += __shfl_xor_sync(0xffffffffu, v[h], o);\n      }\n"
                 "      if (g == 0) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h)\n          if (tok0 + h < cnt) ssb[(size_t)(jst + tok0 + h) * 256 + r16] = v[h];\n      }\n    }\n#endif\n")
    a = rep(a, old, new)
    s0 = "      int qv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n      float ps[8];"
    s1 = "      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n      CTA_SYNC();  // red is reused by the next token\n"
    assert a.count(s0) == 1 and a.count(s1) == 1
    i = a.index(s0); j = a.index(s1) + len(s1)
    new = r'''#if OKC_FIN1 & 2
#if !OKC_YBF
#error "OKC_FIN1 needs OKC_YBF"
#endif
      {
        const float* ssb = p.ybuf + (size_t)EMAX * HD / 2;
        const int qs = S.pos[tk * 8 + (tid >> 6)];
        float4 sv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qs >= 0) sv = __ldcg(reinterpret_cast<const float4*>(ssb + (size_t)qs * 256) + (tid & 63));
        int qv[8];
        float wv[8];
        uint4 yr[8];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          qv[sl] = S.pos[tk * 8 + sl];
          wv[sl] = 0.f;
          yr[sl] = make_uint4(0u, 0u, 0u, 0u);
          if (qv[sl] >= 0) {
            wv[sl] = __ldg(p.topk_w + tk * 8 + sl);
            yr[sl] = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          }
        }
        float s = sv.x + sv.y + sv.z + sv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[warp] = s;
        CTA_SYNC();
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {
            const float r = wv[sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
            const uint32_t* yw = reinterpret_cast<const uint32_t*>(&yr[sl]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yw[i]));
              acc[2 * i] += r * f.x; acc[2 * i + 1] += r * f.y;
            }
          }
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
          ow[i] = *reinterpret_cast<uint32_t*>(&v);
        }
        *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
        CTA_SYNC();  // red is reused by the next token
      }
#else
'''
    a = a[:i] + new + a[i:j] + "#endif\n" + a[j:]
    return a

# OKC_BOFF (lambda in percent, integer; 0 = off): phase-B stream-K boundaries offset by each CTA's predicted phase-A lateness.
# The A ranges equalize the A cost table, but measured A stage times by n-tiles are flatter (OKC_WGST, r4: fat CTAs
# end A 20-30 us before skinny ones, deterministically). B boundary c sits at the uniform position minus lambda x the
# true-cost deviation of the A prefix [0, a_c): early-finishing CTAs take more B. Every boundary is a pure function of
# c (all CTAs agree); the fixup-slot owner of a split B unit is found by binary search over c.
def patch_boff(a, T="112, 122, 130, 138, 146, 163, 168, 173"):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_BOFF\n#define OKC_BOFF 0\n#endif\n#ifndef OKC_WGST\n#define OKC_WGST " + T + "\n#endif\n")
    a = rep(a, "  uint32_t wtotB;     // WGS: phase-B total cost\n", "  uint32_t wtotB;     // WGS: phase-B total cost\n  int wcpT[JMAX + 1]; // BOFF: phase-A true-cost prefix per job\n  uint32_t wtotT;\n")
    a = rep(a, "constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);",
            "constexpr int WGS_CT[8] = {OKC_WGST};\nconstexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB), WGS_PT = wgs_pack8(WGS_CT);")
    a = rep(a, "  uint32_t acc = 0, accB = 0;\n  for (int j = 0; j < nj; ++j) {\n    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;\n",
            "  uint32_t acc = 0, accB = 0, accT = 0;\n  for (int j = 0; j < nj; ++j) {\n    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB; S.wcpT[j] = (int)accT;\n"
            "    accT += ((uint32_t)((WGS_PT >> (8 * (((S.jC[j] + 7) >> 3) - 1))) & 255ull) * 16u + OKC_WGSU) << 5;\n")
    a = rep(a, "  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;\n  S.wtot = acc; S.wtotB = accB;\n",
            "  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB; S.wcpT[nj] = (int)accT;\n  S.wtot = acc; S.wtotB = accB; S.wtotT = accT;\n")
    helpers = r'''
#if OKC_BOFF
// true cost of phase-A stages [0, g) (the unit cost sits before its stage 0, as in wgs_stage_at)
__device__ __forceinline__ float wgs_ftrue(const Smem& S, int nj, int g) {
  const int j = g >> 9;
  if (j >= nj) return (float)S.wtotT;
  const int r = g & 511, un = r >> 4, st = r & 15;
  const uint32_t cs = (uint32_t)((WGS_PT >> (8 * (((S.jC[j] + 7) >> 3) - 1))) & 255ull), U = (uint32_t)OKC_WGSU;
  return (float)((uint32_t)S.wcpT[j] + (uint32_t)un * (16u * cs + U) + (st ? U + (uint32_t)st * cs : 0u));
}
// phase-B boundary of CTA c (first stage of its B range)
__device__ __forceinline__ int wgs_bbound(const Smem& S, int nj, int c, int G) {
  if (c <= 0) return 0;
  if (c >= G) return nj << 8;
  const int ac = wgs_stage_at(S, nj, 0, S.wtot * (uint32_t)c / (uint32_t)G);
  const float fc = (float)c / (float)G;
  float x = fc * (float)S.wtotB - (0.01f * (float)(OKC_BOFF)) * (wgs_ftrue(S, nj, ac) - fc * (float)S.wtotT);
  x = fminf(fmaxf(x, 0.f), (float)S.wtotB - 1.f);
  return wgs_stage_at(S, nj, 1, (uint32_t)x);
}
#endif
'''
    anchor = "// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same\n"
    a = rep(a, anchor, helpers + anchor)
    a = rep(a, "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n",
            "#if OKC_BOFF\n  S.wgr[2] = wgs_bbound(S, nj, blockIdx.x, gridDim.x);\n  S.wgr[3] = max(S.wgr[2], wgs_bbound(S, nj, blockIdx.x + 1, gridDim.x));\n#else\n"
            "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n#endif\n")
    old = "__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {\n"
    new = old + ("#if OKC_BOFF\n  if (pb) {  // largest c with bbound(c) <= s\n    const int nj = S.njF + S.njS;\n    int lo = 0, hi = G;\n"
                 "    while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (wgs_bbound(S, nj, mid, G) <= s) lo = mid; else hi = mid; }\n    return lo;\n  }\n#endif\n")
    a = rep(a, old, new)
    return a
