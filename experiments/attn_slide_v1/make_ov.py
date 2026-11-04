"""attn_slide_v1 overlap variants on top of bx16 -> dev/modal/_slide/<kind>.py.
ov: q_len-1 plain builds (MODE 1, batched sliding) process two landed tiles per iteration (one 128-key score block per
warp: 2x the independent QK / exp / PV work per dependency chain), 4 stages, two tiles in flight while a pair computes."""
import pathlib, sys
H = pathlib.Path(__file__).resolve().parent
ROOT = H.parents[1]
SRC = (H / 'submission_bx16.py').read_text()
OUT = ROOT / 'dev/modal/_slide'

def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)

LOOP_HEAD = """  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);"""

PAIR_LOOP = r"""  if constexpr (PAIR) {
  // tiles i, i + 1 (both landed) as one 128-key block per warp: keys 32 wsub + [0, 32) of each tile
  auto tbody = [&](auto ntl_c, const int i) {
    constexpr int NTL = decltype(ntl_c)::value;
    float s[4 * NTL][4];
    uint32_t kw[4 * NTL][4];
#pragma unroll
    for (int u = 0; u < NTL; ++u) {
      const uint8_t* K = S.k + ((i + u) % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int key = 32 * wsub + 8 * nt + g;
        const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
        kw[4 * u + nt][0] = kb.x; kw[4 * u + nt][1] = kb.y; kw[4 * u + nt][2] = kb.z; kw[4 * u + nt][3] = kb.w;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 4 * NTL; ++nt) {
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
#pragma unroll
    for (int u = 0; u < NTL; ++u) {
      const int t0 = start + (i + u) * TILE;
      const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
      if (!interior) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
            if (pos >= end) s[4 * u + nt][e] = -INFINITY;
          }
      }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4 * NTL; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4 * NTL; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
      for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
#pragma unroll
      for (int u = 0; u < NTL; ++u) {
        const uint8_t* V = S.v + ((i + u) % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t b0[8], b1[8];
          load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
          const uint32_t a0 = pack_h2(s[4 * u + 2 * ks][0], s[4 * u + 2 * ks][1]);
          const uint32_t a2 = pack_h2(s[4 * u + 2 * ks + 1][0], s[4 * u + 2 * ks + 1][1]);
#pragma unroll
          for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
        }
      }
    }
  };
  // groups: prologue tiles 0..2 (one each); pair 0 waits for tiles 0, 1 (tile 2 may pend) and issues tile 3 into the
  // q staging stage; pair i >= 2 waits for everything (tiles i, i + 1) and refills the stages of tiles i - 2, i - 1
  for (int i = 0; i < ntiles; i += 2) {
    if (i == 0) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();
    if (i > 0 && i + 2 < ntiles) issue_kv(i + 2, true);
    if (i + 3 < ntiles) issue_kv(i + 3, true);
    issue_idx(i + 6);
    issue_idx(i + 7);
    cp_commit();
    if (i + 1 < ntiles) tbody(std::integral_constant<int, 2>{}, i);
    else tbody(std::integral_constant<int, 1>{}, i);
  }
  } else
"""

def build(kind):
    s = SRC
    if kind == 'bx16':
        return s
    if kind in ('probe', 'probe8'):
        kind8 = kind == 'probe8'
        sys.path.insert(0, str(H))
        import make_slide as MS
        s = MS.stamps(s)
        s = rep(s, '#define SPIN_CHECK', '#define STAMPV(k, v) do { if (p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t) : "r"(__float_as_uint(v))); \\\n  p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k)] = _t; } } while (0)\n#define SPIN_CHECK')
        s = rep(s, '''      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
''', '''      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
    STAMPV(9, __uint_as_float(kw[3][3]));
''')
        s = rep(s, '''      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    if (!interior) {''', '''      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    STAMPV(10, s[3][1] + s[0][0]);
    if (!interior) {''')
        s = rep(s, '''    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;''', '''    mx = quad_max(mx);
    STAMPV(11, mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;''')
        s = rep(s, '''#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {   // this warp is done with the stage''', '''#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
        if (ks == 0) STAMPV(12, o[0][7][1]);
      }
      STAMPV(13, o[0][7][1] + o[0][0][0]);
    }
    if constexpr (WS) {   // this warp is done with the stage''')
        if kind8:
            s = rep(s, "    wgmma_commit();\n    if constexpr (!WS) {   // the copies for tile i + STAGES - 1",
                    "    wgmma_commit();\n    STAMPV(9, 0.f);\n    if constexpr (!WS) {   // the copies for tile i + STAGES - 1")
            s = rep(s, "    wgmma_wait<0>();\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(s[e]);",
                    "    STAMPV(10, 0.f);\n    wgmma_wait<0>();\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(s[e]);\n    STAMPV(11, s[31] + s[0]);")
            s = rep(s, "           // done before the next tile (no wgmma in flight across the loop edge)\n          wgmma_wait<0>();\n#pragma unroll\n          for (int e = 0; e < 32; ++e) reg_fence(od[e]);",
                    "           // done before the next tile (no wgmma in flight across the loop edge)\n          STAMPV(12, 0.f);\n          wgmma_wait<0>();\n#pragma unroll\n          for (int e = 0; e < 32; ++e) reg_fence(od[e]);\n          STAMPV(13, od[31] + od[0]);")
        return rep(s, 'name=f"okc_attn_slbx16{tag}_90a"', f'name=f"okc_attn_sl{kind}{{tag}}_90a"')
    if kind in ('ov', 'ov8'):
        s = rep(s, 'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
                'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : ((L == 1) ? 4 : 3);   // (q_len 1: the pair loop\'s 4; others use 3)')
        s = rep(s, '  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n  // flat q_len 8',
                '  // q_len 1 plain builds (batched sliding): two tiles per iteration over 4 stages; other per-request builds 3 stages\n'
                '  constexpr bool PAIR = !FLAT && L == 1 && MODE == 1;\n'
                '  constexpr int STAGES = (!FLAT && L == 1) ? (PAIR ? 4 : 3) : Cfg<L, FLAT>::STAGES;\n  // flat q_len 8')
        s = rep(s, LOOP_HEAD, PAIR_LOOP + LOOP_HEAD)
        if kind == 'dpv':
            pass
        if kind == 'ov8':   # also 8 splits for decode-16 (4 tiles per CTA = 2 pairs)
            s = rep(s, "(16, 1, True, 1): dict(n_splits=16,", "(16, 1, True, 1): dict(n_splits=8,")
    elif kind in ('dpv', 'dpv4'):
        if kind == 'dpv4':
            s = rep(s, 'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
                    'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : ((L == 1) ? 3 : 4);')
        if True:   # per-request q_len 8: the PV wgmma stays in flight into the next tile (its QK issue, copies,
            # V^T transpose); the post-transpose wgmma_wait<0> retires it before P / o are touched
            s = rep(s, '           // done before the next tile (no wgmma in flight across the loop edge)\n          wgmma_wait<0>();',
                    '           // flat: done before the next tile; per-request: retired by the next tile\'s post-transpose wait\n          if constexpr (FLAT) wgmma_wait<0>();')
    else:
        raise SystemExit(kind)
    return rep(s, 'name=f"okc_attn_slbx16{tag}_90a"', f'name=f"okc_attn_sl{kind}{{tag}}_90a"')

if __name__ == '__main__':
    for k in sys.argv[1:]:
        (OUT / f'{k}.py').write_text(build(k)); print(k)
