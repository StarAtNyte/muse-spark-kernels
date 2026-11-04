"""Variants of submission_stack.py (parent.py) -> dev/modal/_fresh/<tag>.py."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attn_fresh_v1/parent.py').read_text()
OUT = ROOT / 'dev/modal/_fresh'
OUT.mkdir(exist_ok=True)

def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)

def stamps(s):
    s = rep(s, '''  const int pair = blockIdx.z;
  if (!FLAT) {''', '''  const int pair = blockIdx.z;
  if (FLAT && p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t));
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); p.dbg[blockIdx.x * 16 + 0] = _t; p.dbg[blockIdx.x * 16 + 15] = sm; }
  if (!FLAT) {''')
    s = rep(s, '''      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }''', '''      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
        if (R == 2 && p.dbg && threadIdx.x == 0 && k < 8) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t));
          p.dbg[blockIdx.x * 16 + 4 + k] = _t; p.dbg[blockIdx.x * 16 + 12 + (k & 1)] = s1 - s0; }
      }''')
    s = rep(s, '''      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }''', '''      run_segments(std::integral_constant<int, 2>());
      if (p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t));
        p.dbg[blockIdx.x * 16 + 1] = _t; p.dbg[blockIdx.x * 16 + 3] = nseg; }
      if constexpr (L == 8) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
      if (p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t));
        p.dbg[blockIdx.x * 16 + 2] = _t; }''')
    s = rep(s, '''  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;''', '''  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  if (FLAT && p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); p.dbg[blockIdx.x * 16 + 14] = _t; }
  bool do_red = true;''')
    return s

def st4(s):
    return rep(s, 'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;', 'static constexpr int STAGES = F ? ((L == 1) ? 4 : 4) : 3;')

def name(s, tag):
    return rep(s, 'name=f"okc_attn_b1fstack{tag}_90a"', f'name=f"okc_attn_fr{tag}{{tag}}_90a"')

HELPER = r'''
// Flat grids: the second-dispatched CTA on each SM (blockIdx >= G/2) streams slower than the first (the warp schedulers
// favour the older CTA; measured 1-3% lower rate), so its share is cut by WAVE_D permille and the first wave's grown by the same.
template <int L>
__device__ __forceinline__ int64_t share_bound(int c, int G, int64_t T) {
  constexpr int64_t D = (L == 8) ? WAVE_D8 : WAVE_D1;
  const int H = G >> 1;
  const int64_t w = c <= H ? (int64_t)c * (1000 + D) : (int64_t)H * (1000 + D) + (int64_t)(c - H) * (1000 - D);
  const int64_t wt = (int64_t)H * (1000 + D) + (int64_t)(G - H) * (1000 - D);
  return T * w / wt;
}
__device__ __forceinline__ int fdiv_i(int x, int d) {'''

def wave(d1, d8, segk=None):
    def f(s):
        s = rep(s, '__device__ __forceinline__ int fdiv_i(int x, int d) {',
                f'#define WAVE_D1 {d1}\n#define WAVE_D8 {d8}\n' + HELPER)
        s = rep(s, 'int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);',
                'int csg = (int)share_bound<L>(blockIdx.x, gridDim.x, Tg), ceg = (int)share_bound<L>(blockIdx.x + 1, gridDim.x, Tg);')
        s = rep(s, 'int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);',
                'int csf = (int)share_bound<L>(cf, Gf, Tf), cef = (int)share_bound<L>(cf + 1, Gf, Tf);')
        s = rep(s, '''    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);''', '''    cs = (int)share_bound<L>(cpart, G, T);
    ce = (int)share_bound<L>(cpart + 1, G, T);''')
        if segk:
            s = rep(s, 'constexpr int SEGKF = (L == 8) ? 600 : 400;', f'constexpr int SEGKF = (L == 8) ? {segk[1]} : {segk[0]};', 2)
            s = rep(s, 'constexpr int SEGK = (L == 8) ? 600 : 400;', f'constexpr int SEGK = (L == 8) ? {segk[1]} : {segk[0]};')
        return s
    return f

V = {'base': lambda s: s, 'stamp': stamps, 'st4': st4, 'wA': wave(6, 10), 'wB': wave(6, 16), 'wC': wave(6, 16, (450, 500)), 'wD': wave(9, 22), 'stampwB': lambda s: stamps(wave(6, 16)(s))}

# ---------------- pair pool: shares 2k (CTA k) and 2k+1 (CTA k + G/2) split 2*POOL_D tiles at their boundary ----------------
def pool(D, ahead=2):
    def f(s):
        # shared-memory fields
        s = rep(s, '''  int pnseg, pchunk;''', '''  int pnseg, pchunk;
  int pstop[8];             // pair pool: stage holds no tile (the producer's claims ran out)
  int ppos[2];              // pair pool: the producers' current claimed tile (double buffered)
  int pool_k, pool_dir, pool_lo;   // pair pool of this CTA's pool segment (-1: none), side, first pool key''')
        # segment signature
        s = rep(s, '''                                        const bool skip_init_ = false) {''', '''                                        const bool skip_init_ = false, const int pool_k_ = -1, const int pool_dir_ = 0,
                                        const int pool_lo_ = 0) {''')
        s = rep(s, '''  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)''',
'''  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)
  // pair pool (flat only): after its own ntiles, the CTA claims tiles of the 2*POOL_D-tile pool it shares with its partner
  // share, one atomic per tile on the pair's packed word (left count | right count << 16): exactly the first 2*POOL_D claims
  // succeed; the left share takes pool tiles from the pool's first key up, the right one from its last key down
  constexpr int POOL_N = FLAT ? 2 * POOL_D : 0;
  const int pool_k = __shfl_sync(0xffffffffu, pool_k_, 0), pool_dir = __shfl_sync(0xffffffffu, pool_dir_, 0),
            pool_lo = __shfl_sync(0xffffffffu, pool_lo_, 0);
  const bool pool_on = FLAT && pool_k >= 0;
  const int nit_all = ntiles + (pool_on ? POOL_N + 1 : 0);
  int* const pool_word = p.ws_cnt + 31000 + (pool_k < 0 ? 0 : pool_k);
  int npool = 0;   // pool tiles this CTA processed (consumers)''')
        # producer loop
        s = rep(s, '''      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];''', f'''      int pclaim = 0;
      for (int x = 0; x < nit_all; ++x) {{
        const int st = (tb + x) % STAGES;
        if (pool_on && q == 0 && x == max(0, ntiles - {ahead})) pclaim = atomicAdd(pool_word, pool_dir ? 0x10000 : 1);
        if (pool_on && x >= ntiles) {{
          if (q == 0) {{
            const int old = pclaim;
            const int l = old & 0xffff, r = old >> 16;
            int pos = -1;
            if (l + r < POOL_N) {{
              pos = pool_dir ? pool_lo + TILE * (POOL_N - 1 - r) : pool_lo + TILE * l;
              pclaim = atomicAdd(pool_word, pool_dir ? 0x10000 : 1);
            }} else if (l + r == POOL_N + 1) {{
              *pool_word = 0;   // the pair's last claim: both shares are done with the word (next launch starts at 0)
            }}
            S.ppos[x & 1] = pos;
          }}
          asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");
          const int pos0 = S.ppos[x & 1];
          if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
          if (pos0 < 0) {{
            if (q == 0) S.pstop[st] = 1;
            mbar_arrive(&S.full[st]);
            break;
          }}
          int pp[PK];
#pragma unroll
          for (int k = 0; k < PK; ++k) pp[k] = __ldg(ptr + ((pos0 + r0 + PR * k) >> p.lp));
          const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
          for (int k = 0; k < PK; ++k) {{
            const int r = r0 + PR * k;
            const int pos = pos0 + r;
            const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
            if (p.off32) {{
              const uint32_t off = ((((uint32_t)pp[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }} else {{
              const int64_t off = (((int64_t)pp[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }}
          }}
          cp_async_mbar_arrive(&S.full[st]);
          continue;
        }}
        int pgn[PK];''')
        # L == 1 consumer loop
        s = rep(s, '''  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
    } else {''', '''  for (int i = 0; i < nit_all; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      if (i >= ntiles) {   // pool tile, or the producers' stop mark
        if (S.pstop[(tb + i) % STAGES]) break;
        ++npool;
      }
    } else {''')
        s = rep(s, '''    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile''', '''    const bool interior = (i >= ntiles) || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));   // (pool tiles: cached keys only)
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile''')
        # L == 8 consumer loop
        s = rep(s, '''  const int nit = ntiles;
  constexpr int UNR8''', '''  const int nit = nit_all;
  constexpr int UNR8''')
        s = rep(s, '''        mbar_wait_full(&S.full[st_c], ph_c);
        patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni''', '''        mbar_wait_full(&S.full[st_c], ph_c);
        if (i >= ntiles) {   // pool tile, or the producers' stop mark
          if (S.pstop[st_c]) break;
          ++npool;
        }
        patched = ((uint32_t)i - r_lo > r_span) && (i < ntiles) && (i >= i_new) && ws_patch_ni''')
        s = rep(s, '''    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);''', '''    const bool common_t = WS && (((uint32_t)i - r_lo <= r_span) || i >= ntiles);''')
        # merge key count (last-CTA mode)
        s = rep(s, '''  if (tid == 0) {
    const int mine = end - start;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");''', '''  if (tid == 0) {
    const int mine = end - start + TILE * npool;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");''')
        # share index: CTA c < G/2 -> share 2c, CTA c + G/2 -> share 2c + 1
        s = rep(s, '''      const int cf = blockIdx.x, Gf = gridDim.x;''', '''      const int Gf = gridDim.x, cf = pool_share(blockIdx.x, Gf);''')
        s = rep(s, '''int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);''',
                '''const int sg = pool_share(blockIdx.x, gridDim.x);
        int csg = (int)((int64_t)sg * Tg / gridDim.x), ceg = (int)((int64_t)(sg + 1) * Tg / gridDim.x);''')
        s = rep(s, '''  const int cpart = c;''', '''  const int cpart = pool_share(c, G);''')
        s = rep(s, '__device__ __forceinline__ int fdiv_i(int x, int d) {', f'''#define POOL_D {D}
// pair pool: the share index of CTA c (an even grid pairs CTA c < G/2, dispatched first, with CTA c + G/2 on another SM)
__device__ __forceinline__ int pool_share(int c, int G) {{
  if (POOL_D == 0 || (G & 1)) return c;
  const int H = G >> 1;
  return c < H ? 2 * c : 2 * (c - H) + 1;
}}
__device__ __forceinline__ int fdiv_i(int x, int d) {{''')
        # pool decision after the segment list
        s = rep(s, '''  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  const int nseg = S.nseg;''', '''  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  if (threadIdx.x == 0) {
    // pair pool: full layers, even grids. Pair k = shares 2k (left) and 2k+1 (right); both CTAs derive the same pool from
    // the three share bounds. The left share's last segment and the right share's only segment must be in one request,
    // own at least a tile each, and the pool must hold cached keys only (below seq: no new rows, no causal mask)
    for (int st = 0; st < 8; ++st) S.pstop[st] = 0;
    S.pool_k = -1;
    const int sh = pool_share(c, G);
    if (POOL_D > 0 && !p.window && !(G & 1) && S.nseg > 0) {
      auto vmp = [&](long long v) -> int {
        int lo_b = 0, hi_b = p.B - 1;
        while (lo_b < hi_b) {
          const int mid = (lo_b + hi_b + 1) >> 1;
          if ((long long)S.cum[mid] + (long long)mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
        }
        return (int)min(v - (long long)lo_b * SEGK, (long long)S.cum[lo_b + 1]);
      };
      const int k = sh >> 1;
      const int r0 = vmp((long long)(2 * k) * T / G), r1 = vmp((long long)(2 * k + 1) * T / G), r2 = vmp((long long)(2 * k + 2) * T / G);
      int bq = 0;
      while (bq + 1 < p.B && S.cum[bq + 1] < r1) ++bq;   // r1 in (cum[bq], cum[bq + 1]]
      const int cb = S.cum[bq], nb = S.cum[bq + 1] - cb;
      const int s0l = max(r0, cb) - cb, el = r1 - cb, er = r2 - cb;
      const int nown = (el - s0l) / TILE - POOL_D;
      const int plo = s0l + TILE * nown, phi = plo + 2 * POOL_D * TILE;
      if (r1 > cb && r1 < S.cum[bq + 1] && r2 <= S.cum[bq + 1] && nown >= 1 && phi + TILE <= er && phi <= nb - L) {
        const int n = S.nseg;
        if (!(sh & 1) && S.seg[n - 1][0] == bq) {
          S.seg[n - 1][2] = plo; S.pool_k = k; S.pool_dir = 0; S.pool_lo = plo;
        } else if ((sh & 1) && n == 1 && S.seg[0][0] == bq) {
          S.seg[0][1] = phi; S.pool_k = k; S.pool_dir = 1; S.pool_lo = plo;
        }
      }
    }
  }
  __syncthreads();
  const int nseg = S.nseg;''')
        # pass the pool to the segment
        s = rep(s, '''        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);''', '''        const bool pseg_ = S.pool_k >= 0 && (S.pool_dir ? k == 0 : k == nseg - 1);
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0, pseg_ ? S.pool_k : -1, S.pool_dir, S.pool_lo);''')
        return s
    return f

def pool2(D, l8=True, off=False, ahead=3):
    def f(s):
        # shared-memory fields
        s = rep(s, '''  int pnseg, pchunk;''', '''  int pnseg, pchunk;
  int pstop[8];             // pair pool: stage holds no tile (the producer's claims ran out)
  int ppos[2];              // pair pool: the producers' current claimed tile (double buffered)
  int pool_k, pool_dir, pool_lo;   // pair pool of this CTA's pool segment (-1: none), side, first pool key
  int pool_i0;              // pair pool: first pool tile index of the running segment (consumers, q_len 8)''')
        # segment signature
        s = rep(s, '''                                        const bool skip_init_ = false) {''', '''                                        const bool skip_init_ = false, const int pool_k_ = -1, const int pool_dir_ = 0,
                                        const int pool_lo_ = 0) {''')
        s = rep(s, '''  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)''',
'''  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)
  // pair pool (flat only): after its own ntiles, the CTA claims tiles of the 2*POOL_D-tile pool it shares with its partner
  // share, one atomic per tile on the pair's packed word (left count | right count << 16): exactly the first 2*POOL_D claims
  // succeed; the left share takes pool tiles from the pool's first key up, the right one from its last key down
  constexpr int POOL_N = FLAT && (L == 1 || POOL_L8) ? 2 * POOL_D : 0;   // pool tiles (claimed in units of 2)
  const int pool_k = __shfl_sync(0xffffffffu, pool_k_, 0), pool_dir = __shfl_sync(0xffffffffu, pool_dir_, 0),
            pool_lo = __shfl_sync(0xffffffffu, pool_lo_, 0);
  const bool pool_on = POOL_N > 0 && pool_k >= 0;
  const int nit_all = ntiles + (pool_on ? POOL_N + 1 : 0);
  int* const pool_word = p.ws_cnt + 31000 + (pool_k < 0 ? 0 : pool_k);
  int npool = 0;   // pool tiles this CTA processed (consumers)''')
        # producer loop
        s = rep(s, '''      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];''', f'''      int pclaim = 0, pu = -1;
      int pp[2 * PK];
      for (int x = 0; x < nit_all; ++x) {{
        const int st = (tb + x) % STAGES;
        if (pool_on && q == 0 && x == max(0, ntiles - {ahead})) pclaim = atomicAdd(pool_word, pool_dir ? 0x10000 : 1);
        if (pool_on && x >= ntiles) {{
          const int xu = x - ntiles;
          if ((xu & 1) == 0) {{   // a new unit of two tiles: decode the claim issued earlier, issue the next one
            if (q == 0) {{
              const int old = pclaim;
              const int l = old & 0xffff, r = old >> 16;
              int pos = -1;
              if (l + r < POOL_D) {{
                pos = pool_dir ? pool_lo + 2 * TILE * (POOL_D - 1 - r) : pool_lo + 2 * TILE * l;
                pclaim = atomicAdd(pool_word, pool_dir ? 0x10000 : 1);
              }} else if (l + r == POOL_D + 1) {{
                *pool_word = 0;   // the pair's last claim: both shares are done with the word (next launch starts at 0)
              }}
              S.ppos[(xu >> 1) & 1] = pos;
            }}
            asm volatile("bar.sync 2, %0;\\n" ::"n"(NP) : "memory");
            pu = S.ppos[(xu >> 1) & 1];
            if (pu >= 0) {{
#pragma unroll
              for (int k = 0; k < 2 * PK; ++k) pp[k] = __ldg(ptr + ((pu + r0 + PR * k) >> p.lp));   // both tiles' pages
            }}
          }}
          if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
          if (pu < 0) {{
            if (q == 0) S.pstop[st] = 1;
            mbar_arrive(&S.full[st]);
            break;
          }}
          const int pos0 = pu + TILE * (xu & 1);
          const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
          for (int k = 0; k < PK; ++k) {{
            const int r = r0 + PR * k;
            const int pos = pos0 + r;
            const int pg_ = (xu & 1) ? pp[PK + k] : pp[k];
            const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
            if (p.off32) {{
              const uint32_t off = ((((uint32_t)pg_ << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }} else {{
              const int64_t off = (((int64_t)pg_ << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }}
          }}
          cp_async_mbar_arrive(&S.full[st]);
          continue;
        }}
        int pgn[PK];''')
        # L == 1 consumer loop
        s = rep(s, '''  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
    } else {''', '''  for (int i = 0; i < nit_all; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      if (i >= ntiles) {   // pool tile, or the producers' stop mark
        if (S.pstop[(tb + i) % STAGES]) break;
        ++npool;
      }
    } else {''')
        s = rep(s, '''    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile''', '''    const bool interior = (i >= ntiles) || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));   // (pool tiles: cached keys only)
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile''')
        # L == 8 consumer loop
        s = rep(s, '''  const int nit = ntiles;
  constexpr int UNR8''', '''  if (tid == 0) S.pool_i0 = pool_on ? ntiles : 0x7fffffff;   // (read after tile 0's barrier; pool tiles have i >= 1)
  const int nit = nit_all;
  constexpr int UNR8''')
        s = rep(s, '''        mbar_wait_full(&S.full[st_c], ph_c);
        patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni''', '''        mbar_wait_full(&S.full[st_c], ph_c);
        if (POOL_N > 0 && i > 0 && i >= ld_shv(&S.pool_i0) && ld_shv(&S.pstop[st_c])) break;   // the producers' stop mark
        patched = ((uint32_t)i - r_lo > r_span) && (POOL_N == 0 || i == 0 || i < ld_shv(&S.pool_i0)) && (i >= i_new) && ws_patch_ni''')
        s = rep(s, '''    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);''', '''    const bool common_t = WS && (((uint32_t)i - r_lo <= r_span) || (POOL_N > 0 && i > 0 && i >= ld_shv(&S.pool_i0)));''')
        # merge key count (last-CTA mode)
        s = rep(s, '''  if (tid == 0) {
    const int mine = end - start;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");''', '''  if (tid == 0) {
    const int mine = end - start + TILE * npool;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");''')
        # share index: CTA c < G/2 -> share 2c, CTA c + G/2 -> share 2c + 1
        s = rep(s, '''      const int cf = blockIdx.x, Gf = gridDim.x;''', '''      const int Gf = gridDim.x, cf = pool_share(blockIdx.x, Gf);''')
        s = rep(s, '''int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);''',
                '''const int sg = pool_share(blockIdx.x, gridDim.x);
        int csg = (int)((int64_t)sg * Tg / gridDim.x), ceg = (int)((int64_t)(sg + 1) * Tg / gridDim.x);''')
        s = rep(s, '''  const int cpart = c;''', '''  const int cpart = pool_share(c, G);''')
        s = rep(s, '__device__ __forceinline__ int fdiv_i(int x, int d) {', f'''#define POOL_D {D}
#define POOL_L8 {int(l8)}
#define POOL_OFF {int(off)}
__device__ __forceinline__ int ld_shv(const int* a) {{ int v; asm volatile("ld.volatile.shared.s32 %0, [%1];" : "=r"(v) : "r"(smem_u32(a))); return v; }}
// pair pool: the share index of CTA c (an even grid pairs CTA c < G/2, dispatched first, with CTA c + G/2 on another SM)
__device__ __forceinline__ int pool_share(int c, int G) {{
  if (POOL_D == 0 || (G & 1)) return c;
  const int H = G >> 1;
  return c < H ? 2 * c : 2 * (c - H) + 1;
}}
__device__ __forceinline__ int fdiv_i(int x, int d) {{''')
        # pool decision after the segment list
        s = rep(s, '''  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  const int nseg = S.nseg;''', '''  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  if (threadIdx.x == 0) {
    // pair pool: full layers, even grids. Pair k = shares 2k (left) and 2k+1 (right); both CTAs derive the same pool from
    // the three share bounds. The left share's last segment and the right share's only segment must be in one request,
    // own at least a tile each, and the pool must hold cached keys only (below seq: no new rows, no causal mask)
    for (int st = 0; st < 8; ++st) S.pstop[st] = 0;
    S.pool_k = -1;
    const int sh = pool_share(c, G);
    if (POOL_D > 0 && !p.window && !(G & 1) && S.nseg > 0) {
      auto vmp = [&](long long v) -> int {
        int lo_b = 0, hi_b = p.B - 1;
        while (lo_b < hi_b) {
          const int mid = (lo_b + hi_b + 1) >> 1;
          if ((long long)S.cum[mid] + (long long)mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
        }
        return (int)min(v - (long long)lo_b * SEGK, (long long)S.cum[lo_b + 1]);
      };
      const int k = sh >> 1;
      const int r0 = vmp((long long)(2 * k) * T / G), r1 = vmp((long long)(2 * k + 1) * T / G), r2 = vmp((long long)(2 * k + 2) * T / G);
      int bq = 0;
      while (bq + 1 < p.B && S.cum[bq + 1] < r1) ++bq;   // r1 in (cum[bq], cum[bq + 1]]
      const int cb = S.cum[bq], nb = S.cum[bq + 1] - cb;
      const int s0l = max(r0, cb) - cb, el = r1 - cb, er = r2 - cb;
      const int nown = (el - s0l) / TILE - POOL_D;
      const int plo = s0l + TILE * nown, phi = plo + 2 * POOL_D * TILE;
      if (!POOL_OFF && r1 > cb && r1 < S.cum[bq + 1] && r2 <= S.cum[bq + 1] && nown >= 1 && phi + TILE <= er && phi <= nb - L) {
        const int n = S.nseg;
        if (!(sh & 1) && S.seg[n - 1][0] == bq) {
          S.seg[n - 1][2] = plo; S.pool_k = k; S.pool_dir = 0; S.pool_lo = plo;
        } else if ((sh & 1) && n == 1 && S.seg[0][0] == bq) {
          S.seg[0][1] = phi; S.pool_k = k; S.pool_dir = 1; S.pool_lo = plo;
        }
      }
    }
  }
  __syncthreads();
  const int nseg = S.nseg;''')
        # pass the pool to the segment
        s = rep(s, '''        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);''', '''        const bool pseg_ = S.pool_k >= 0 && (S.pool_dir ? k == 0 : k == nseg - 1);
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0, pseg_ ? S.pool_k : -1, S.pool_dir, S.pool_lo);''')
        return s
    return f

V.update({'q6': pool2(6), 'q6off': pool2(6, off=True), 'q6l1': pool2(6, l8=False), 'q4': pool2(4)})
V.update({'pool4': pool(4), 'pool6': pool(6), 'pool8': pool(8), 'stamppool6': lambda s: pool(6)(stamps(s))})

if __name__ == "__main__":
    import sys
    for tag in (sys.argv[1:] or V):
        (OUT / f'{tag}.py').write_text(name(V[tag](SRC), tag))
        print('wrote', tag)
