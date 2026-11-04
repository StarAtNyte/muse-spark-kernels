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
if __name__ == '__main__':
    import sys
    for tag in (sys.argv[1:] or V):
        (OUT / f'{tag}.py').write_text(name(V[tag](SRC), tag))
        print('wrote', tag)
