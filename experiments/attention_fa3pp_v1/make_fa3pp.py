"""Build the FA3-style pipelined q_len-8 flat (full-layer, B > 1) attention variants from the readable parent.

Change (only the flat q_len 8 route, Cfg<8, true>): software-pipelined consumer loop per warpgroup -- tile i+1's
QK^T wgmma is issued before tile i's softmax, and tile i's PV wgmma is left in flight across the loop edge (waited
by the next tile's wgmma wait) -- plus a configurable CTAs-per-SM / stage count for that route, since the second
score buffer needs ~32 more registers than the 2-CTA/SM 104-register budget allows.
All other routes are byte-identical in behaviour.

Writes the candidate (env defaults) to experiments/attention_fa3pp_v1/submission.py and screen variants to
dev/modal/_fa3pp/<tag>.py.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attention_readable_fp32_path_v1/submission.py').read_text()
OUT = ROOT / 'dev/modal/_fa3pp'
OUT.mkdir(exist_ok=True)

WO_HELPER = r"""// write-only accumulators, scale-d = 0: the previous use of these registers (the masked scores of an earlier tile)
// does not feed this wgmma (FA3PP keeps two score buffers live across the loop edge)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db), "r"(0));
}
"""
WO_ANCHOR = '// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)\n'


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    return s.replace(a, b)


NEW_LOOP = r'''
  if constexpr (WS && FA3PP) {
  // ---- flat q_len 8, FA3-style intra-warpgroup pipeline. Per tile i:
  //   acquire tile i+1, issue QK^T(i+1) -> transpose V(i) -> wait QK^T(i) (and PV(i-1)) -> softmax(i) -> issue PV(i)
  // so the tensor cores run QK^T(i+1) during softmax(i), and PV(i) during the next tile's acquire / transpose.
  // Commit order QK(i), PV(i-1), QK(i+1): wgmma_wait<1> after QK(i+1) retires both QK(i) and PV(i-1), which frees
  // P (pa), settles o / lacc for a rare rescale, and frees V^T buffer (i & 1) for tile i+2.
  const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
  float sB_[FA3LA ? 32 : 1];
  auto acquire = [&](int i, int stx, uint32_t phx) {
    mbar_wait_full(&S.full[stx], phx);
    // the block's new rows into the stage, inline: a call here would sit inside PV(i-1)'s wgmma pipeline stage
    // (ptxas then serializes every wgmma, C7514)
    bool patched = false;
    if (((uint32_t)i - r_lo > r_span) && (i >= i_new)) {
      const int t0 = start + i * TILE;
      const int lo = max(t0, seq), hi = min(t0 + TILE, end);
      if (lo < hi) {   // same for every consumer thread
        for (int e = tid; e < (hi - lo) * 8; e += NT) {
          const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + stx * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          *reinterpret_cast<uint4*>(S.v + stx * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        fence_proxy_async();
        patched = true;
      }
    }
    if (patched || i == 0) csync<WS, NT>();
  };
  auto issue_qk = [&](float* sx, int stx) {
    const uint32_t kl = klo0 + (uint32_t)(stx * (STAGE_BYTES >> 4));
    const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
    const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
    wgmma_fence();
    wgmma_m64n64k32_e4m3_ss_z(sx, qdesc0, kd);
    wgmma_m64n64k32_e4m3_ss(sx, qd2, kd2, 1);
    wgmma_commit();
  };
  auto tile = [&](float* s, float* sn, const int i, auto next_tag) {
    constexpr bool has_next = decltype(next_tag)::value;
    const int st = st_c;
    int st_n = st_c + 1;
    uint32_t ph_n = ph_c;
    if (st_n == STAGES) { st_n = 0; ph_n ^= 1u; }
    if constexpr (FA3LA) {   // look-ahead: QK^T(i+1) now, in flight during softmax(i)
      if constexpr (has_next) {
        acquire(i + 1, st_n, ph_n);
        issue_qk(sn, st_n);
      }
    } else {   // no look-ahead (one score buffer): only PV(i-1) overlaps this tile's acquire / QK^T / transpose
      if (i > 0) acquire(i, st, ph_c);
      issue_qk(s, st);
    }
    // this head's V tile i -> V^T buffer (i & 1) (FP8, 72 x 64, keys in P-register order), while QK^T(i+1) and PV(i-1) run.
    // PV(i-2) read this buffer; it retired at tile i-1's wait.
    {
      const uint32_t vsrc = vsrc0 + st * STAGE_BYTES;
      uint8_t* vtb = vtb_c;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (vld_off0 ^ 32u) : vld_off0), r0, r1, r2, r3);
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;
        }
      }
    }
    if constexpr (FA3LA && has_next) wgmma_wait<1>(); else wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    {
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    mbar_arrive(&S.empty[st]);   // K (QK^T wgmma, retired) and V (ldmatrix, in registers above) of the stage consumed

    const bool common_t = ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = common_t || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));
    // the mask never writes the scores (the wgmma's accumulator registers: ptxas then serializes every wgmma of
    // the kernel, C7515): the row maxima read a masked copy x, and P is masked bytewise after packing (keep bits)
    // (no look-ahead: nothing is in flight here, so the mask writes the scores in place)
    float xbuf[FA3LA ? 32 : 1];
    float* const x = FA3LA ? xbuf : s;
    if constexpr (FA3LA) {
#pragma unroll
      for (int e = 0; e < 32; ++e) x[e] = s[e];
    }
    uint32_t keep = 0xffffffffu;
    if (!interior) {
      keep = 0u;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) { x[4 * nt + 2 * hf + e] = -INFINITY; } else keep |= 1u << (4 * nt + 2 * hf + e);
          }
      }
    }
    const bool chk = !first;
    float dlo, dhi;
    if (first) {
      float mlo = -INFINITY, mhi = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        mlo = fmaxf(mlo, fmaxf(x[4 * nt], x[4 * nt + 1]));
        mhi = fmaxf(mhi, fmaxf(x[4 * nt + 2], x[4 * nt + 3]));
      }
      mlo = quad_max(mlo * ssc); mhi = quad_max(mhi * ssc);
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
    }
    auto pack2 = [&]() {
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const float* sv = s + 16 * kk;   // (no look-ahead: x == s, masked)
        pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
        pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
        pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
        pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
      }
      if (FA3LA && !interior) {   // P = 0 at masked keys: element e of pa[kk][w] byte b is s[16kk + 8(w >> 1) + 2(w & 1) + 4(b >> 1) + (b & 1)]
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            uint32_t m = 0u;
#pragma unroll
            for (int bb = 0; bb < 4; ++bb)
              m |= ((keep >> (16 * kk + 8 * (w >> 1) + 2 * (w & 1) + 4 * (bb >> 1) + (bb & 1))) & 1u) ? (0xffu << (8 * bb)) : 0u;
            pa[kk][w] &= m;
          }
      }
    };
    pack2();
    if (chk) {
      // an e4m3 byte above 0x78 (P > 256: a score past its row's offset + 8) sets bit 7 when 0x07 is added
      constexpr uint32_t PB = 0x07070707u;
      const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                           (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
      if (__any_sync(0xffffffffu, ov != 0u)) {
        // rare: offsets up to the tile's row maxima, state rescaled (PV(i-1) retired at the wait above, PV(i) not
        // issued), both halves repacked
        float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          hlo = fmaxf(hlo, fmaxf(x[4 * nt], x[4 * nt + 1]));
          hhi = fmaxf(hhi, fmaxf(x[4 * nt + 2], x[4 * nt + 3]));
        }
        hlo = quad_max(hlo) * ssc;
        hhi = quad_max(hhi) * ssc;
        const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        pack2();
      }
    }
    float* odk = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    fence_proxy_async();   // every warp's V^T for this tile written and visible to the async proxy
    wgsync(wgi);
    wgmma_fence();
    wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
    wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
    wgmma_commit();
    st_c = st_n; ph_c = ph_n;
    vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
    vdesc_c ^= vdesc_x;
  };
  float sA[32];
  float* const sB = FA3LA ? sB_ : sA;
  if (ntiles > 0) {
    acquire(0, st_c, ph_c);
    {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV wgmma also produces the row sums
        // (after tile 0's barrier: the q staging shares this space)
      const int wt = tid & 127;
      *reinterpret_cast<uint2*>(S.kvq + ((0 + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
          make_uint2(0x38383838u, 0x38383838u);
    }
    if constexpr (FA3LA) issue_qk(sA, st_c);
  }
  // (tile counts static per call site: no branch decides which wgmma groups are in flight)
  using NX = std::integral_constant<bool, true>;
  using LS = std::integral_constant<bool, false>;
  int i = 0;
  if constexpr (!FA3LA) {   // one score buffer, one copy of the tile body
#pragma unroll 1
    for (; i < ntiles; ++i) tile(sA, sA, i, LS());
  } else {
#pragma unroll 1
  for (; i + 2 < ntiles; i += 2) {
    tile(sA, sB, i, NX());
    tile(sB, sA, i + 1, NX());
  }
  if (i + 1 < ntiles) {
    tile(sA, sB, i, NX());
    tile(sB, sA, i + 1, LS());
  } else {
    tile(sA, sB, i, LS());
  }
  }
  } else {
'''

def build(tag, pp, minb, stages, ctas, out=None, la=True):
    s = SRC.replace('okc_attn_v298', f'okc_attn_fa3pp_{tag}_')
    # route config: flat q_len 8 only
    s = rep(s, '  static constexpr int MINB = 2;   // CTAs per SM the registers must allow\n',
            '  // flat q_len 8 (FA3PP route): CTAs per SM the registers must allow; all other routes 2\n'
            f'  static constexpr int MINB = (F && L == 8) ? {minb} : 2;   // CTAs per SM the registers must allow\n')
    s = rep(s, '  static constexpr int BASEREGS = (65536 / (2 * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (2 * NTX)) / 8 * 8;',
            '  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;')
    s = rep(s, '  static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
            f'  static constexpr int STAGES = F ? ((L == 1) ? 3 : {stages}) : 3;')
    s = rep(s, 'constexpr int TILE = 64;\n',
            'constexpr int TILE = 64;\n'
            f'constexpr bool FA3PP = {"true" if pp else "false"};   // flat q_len 8: pipelined consumer loop\n'
            f'constexpr bool FA3LA = {"true" if la else "false"};   // FA3PP: QK^T(i+1) issued before softmax(i)\n')
    s = rep(s, WO_ANCHOR, WO_HELPER + WO_ANCHOR)
    # the loop
    anchor = '  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup\'s claimed tile, and how many it has taken\n'
    s = rep(s, anchor, NEW_LOOP + anchor)
    tail = ('      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }\n'
            '      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);\n'
            '      vdesc_c ^= vdesc_x;\n    \n  }\n  PROF_DUMP;\n  pdl_launch_dependents();\n  \n  wgmma_wait<0>();')
    s = rep(s, tail, tail.replace('  }\n  PROF_DUMP;', '  }\n  }   // (FA3PP)\n  PROF_DUMP;', 1))
    # host plan: flat q_len 8 grid = ctas CTAs per SM
    s = rep(s, '        return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)',
            f'        flat_target = {ctas} * sms if L == 8 else target   # flat q_len 8: CTAs per SM of its route\n'
            '        return dict(n_splits=max(1, flat_target // pairs), flat=1, cs=0, ncl=1, nc=1)')
    (out or OUT / f'{tag}.py').write_text(s)


if __name__ == '__main__':
    base = SRC.replace('okc_attn_v298', 'okc_attn_fa3pp_base_')
    (OUT / 'base.py').write_text(base)
    build('ctl', False, 2, 4, 2)
    build('pp1s8', True, 1, 8, 1)
    build('m1s8', False, 1, 8, 1)
    build('np2s4', True, 2, 4, 2, la=False)
    build('np1s8', True, 1, 8, 1, la=False)
    build('cand', True, 1, 8, 1, out=ROOT / 'experiments/attention_fa3pp_v1/submission.py')
    print(sorted(p.name for p in OUT.iterdir()))
