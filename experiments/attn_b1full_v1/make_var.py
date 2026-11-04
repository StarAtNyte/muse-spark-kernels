"""Variants of the g16b parent for B=1 full-context work -> dev/modal/_b1full/<tag>.py."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attn_b1full_v1/parent.py').read_text()
OUT = ROOT / 'dev/modal/_b1full'
OUT.mkdir(exist_ok=True)

def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)

STAMP_DEF = r'''
#define STAMP(k) do { if (p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); \
  p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k)] = _t; } } while (0)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {'''

def stamps(s):
    s = rep(s, '\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {', STAMP_DEF)
    s = rep(s, '''  const int pair = blockIdx.z;
  if (!FLAT) {''', '''  const int pair = blockIdx.z;
  STAMP(0);
  if (p.dbg && threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 15] = sm; }
  if (!FLAT) {''')
    s = rep(s, '''  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
''', '''  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  STAMP(1);
''')
    s = rep(s, '''    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
''', '''    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  STAMP(2);
''')
    s = rep(s, '''      cp_wait<STAGES - 2>();
      __syncthreads();
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);''', '''      cp_wait<STAGES - 2>();
      __syncthreads();
      if (i == 0) STAMP(3);
      if (i == ntiles - 1) STAMP(9);
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);''')
    s = rep(s, '''  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;''', '''  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  STAMP(4);
  bool do_red = true;''')
    s = rep(s, '''  if (!(mode1 || !mode1)) gpu_fence();
  csync<WS, NT>();
''', '''  if (!(mode1 || !mode1)) gpu_fence();
  csync<WS, NT>();
  STAMP(5);
''')
    # coop: the arrive atomic's return
    s = rep(s, '''      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      ''', '''      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      STAMP(8);
      ''')
    s = rep(s, '''    csync<WS, NT>();
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);''', '''    csync<WS, NT>();
    STAMP(6);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);''')
    s = rep(s, '''    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
''', '''    S.flag = (old + mine == n_keys);
    STAMP(8);
  }
  csync<WS, NT>();
  STAMP(6);
''')
    s = rep(s, '''     // consumers' shared memory is reused by the next segment
   __syncthreads();                   // shared memory is reused by the next segment''', '''     // consumers' shared memory is reused by the next segment
   STAMP(7);
   __syncthreads();                   // shared memory is reused by the next segment''')
    return s

BF_RED_L1 = '''    if constexpr (L == 1) {
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
      for (int c = 0; c < 2; ++c)'''

def bfd(s):
    """BF16 split partials also for the one-request q_len-1 cooperative speculative builds (d1f)."""
    s = rep(s, '''  constexpr bool BFP = (L == 8) && !FLAT;''',
            '''  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 11);''')
    s = rep(s, '''    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)''', BF_RED_L1)
    return s

def tuned(key, val):
    def f(s):
        return rep(s, f"    {key}: dict(nc=8, coop=1, n_splits=252),", f"    {key}: dict({val}),")
    return f

def chain(*fs):
    def g(s):
        for f in fs: s = f(s)
        return s
    return g

V4R_OLD = """      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1).."""
V4R_NEW = """      if constexpr (BFP) {
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
      } else {
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1).."""
V4R_OLD2 = """        acc_red4<BFP>(p, acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    }"""
V4R_NEW2 = """        acc_red4<BFP>(p, acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
      }
    }"""
def v4r(s):
    """q_len-8 bf16 partials as 16-byte v4.bf16x2 atomics (quad transpose) instead of 8-byte v2.bf16x2."""
    return rep(rep(s, V4R_OLD, V4R_NEW), V4R_OLD2, V4R_NEW2)

NC16_OLD = """  if (p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip"""
NC16_NEW = """  if (BF && L == 1 && p.nc == 16) {   // sixteen bf16 copies (one-request q_len 1 full plan): one round trip
    if (act) {
      float4 xa[16], xb[16];
      float lc[16];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 16 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = acc_ld4<BF>(p, acc, 0);
        xb[c] = acc_ld4<BF>(p, acc, 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 16 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 16 + c) * L + j) * 16 + hh) * 64 + part * 8;
        acc_zero4<BF>(p, acc, 0);
        acc_zero4<BF>(p, acc, 1);
        if (part == 0) p.ws_l[((bp * 16 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
""" + NC16_OLD
def nc16(s):
    return rep(s, NC16_OLD, NC16_NEW)

def fbf(s):
    """bf16 partials (v4.bf16x2 after v4r) also for the flat q_len-8 kernels (v16f / v32f)."""
    s = rep(s, "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 11);",
               "  constexpr bool BFP = L == 8 || (!FLAT && (MODE & 11) == 11);")
    s = rep(s, "    sum_copies<L, false>(p, bp, j, hh, part, act, y, lsum);",
               "    sum_copies<L, L == 8>(p, bp, j, hh, part, act, y, lsum);")
    return s

def nh(s):
    """bf16 q_len-1 builds: both key halves add their own state to the workspace (no shared-memory handoff)."""
    return rep(s, """  if (L == 1) {
    // warps 2,3 (second key half) hand their state to warps 0,1""", """  if (L == 1 && !BFP) {
    // warps 2,3 (second key half) hand their state to warps 0,1""")

def bxs(s):
    """bx16 (attn_slide_v1): bf16 partials also in the plain non-speculative q_len-1 builds, d16s at 16 splits."""
    s = rep(s, "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 11);",
               "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 1 || (MODE & 11) == 11);")
    s = rep(s, "    (16, 1, True, 1): dict(n_splits=8, coop=0, nc=1),   # H200 paired sweep: 1.029x on 16-request sliding decode",
               "    (16, 1, True, 1): dict(n_splits=16, coop=0, nc=1),   # with bf16 q_len-1 partials: n 16 beats n 8 by ~9% (attn_slide_v1 r6)")
    return s

def bfs(s):
    """bf16 partials also in the one-request q_len-1 non-cooperative speculative builds (d1s)."""
    return rep(s, "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 1 || (MODE & 11) == 11);",
                  "  constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 1 || (MODE & 11) == 9 || (MODE & 11) == 11);")

def warm(s):
    """Run the nc-1 last-CTA finalize dry (no loads / stores) while the page indices are in flight: warm i-cache."""
    s = rep(s, """      xa[u] = acc_ld4<BF>(p, acc, 0);
      xb[u] = acc_ld4<BF>(p, acc, 1);
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);""",
    """      if (!dry) {
        xa[u] = acc_ld4<BF>(p, acc, 0);
        xb[u] = acc_ld4<BF>(p, acc, 1);
        lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
        graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      } else {
        xa[u] = make_float4(0.f, 0.f, 0.f, 0.f); xb[u] = xa[u]; lc[u] = 0.f; graw[u] = make_uint4(0u, 0u, 0u, 0u);
      }""")
    s = rep(s, """  if (M_CL) cluster_finish<L>(p, S, b, pair, n_keys, chunk, true);
""", """  if (M_CL) cluster_finish<L>(p, S, b, pair, n_keys, chunk, true);
  if (!FLAT && !M_CL && !M_COOP && p.nc == 1) fin_nc1<L, NT, BFP>(p, b, pair, bp, 1.f, 1);
""")
    return s

def relt(s):
    """Non-cooperative ticket: atom.release (no acquire); only the last CTA issues fence.acq_rel before its reads."""
    return rep(s, """    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    
    S.flag = (old + mine == n_keys);""", """    asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    if (old + mine == n_keys) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    S.flag = (old + mine == n_keys);""")

VARIANTS = {
    'base': lambda s: s,
    'stamp': stamps,
    'bfd': bfd,
    'bfd4': chain(bfd, tuned('(1, 1, False, 1)', 'nc=4, coop=1, n_splits=252')),
    'bfdst': chain(bfd, stamps),
    'v4r': v4r,
    'v4rst': chain(v4r, stamps),
    'bv': chain(bfd, v4r),
    'bfd16': chain(bfd, nc16, tuned('(1, 1, False, 1)', 'nc=16, coop=1, n_splits=252')),
    'bfd16r': chain(bfd, tuned('(1, 1, False, 1)', 'nc=16, coop=1, n_splits=252')),
    'bv16': chain(bfd, v4r, nc16, tuned('(1, 1, False, 1)', 'nc=16, coop=1, n_splits=252')),
    'bvnc': chain(bfd, v4r, nc16, tuned('(1, 1, False, 1)', 'nc=16, coop=0, n_splits=252')),
    'bvq16': chain(bfd, v4r, lambda s: rep(s, "    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),",
        "    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),\n    (1, 8, False, 1): dict(nc=16, coop=1, n_splits=252),")),
    'bvf': chain(bfd, v4r, fbf),
    'bvnh': chain(bfd, v4r, nh),
    'bvst': chain(bfd, v4r, stamps),
    'stack': chain(bfd, v4r, bxs),
    's2a': chain(bfd, v4r, bxs, bfs),
    's2b': chain(bfd, v4r, bxs, bfs, warm),
    's2c': chain(bfd, v4r, bxs, bfs, warm, relt),
    'bfdnc': chain(bfd, nc16, tuned('(1, 1, False, 1)', 'nc=16, coop=0, n_splits=252')),
}

def write(tag, s):
    s = rep(s, 'name=f"okc_attn_vfg16bs{tag}_90a"', f'name=f"okc_attn_b1f{tag}{{tag}}_90a"')
    (OUT / f'{tag}.py').write_text(s)

if __name__ == '__main__':
    for t in (sys.argv[1:] or VARIANTS):
        write(t, VARIANTS[t](SRC))
        print('wrote', t)
