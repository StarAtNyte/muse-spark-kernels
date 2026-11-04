
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
