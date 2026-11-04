"""4-warpgroup q_len-8 per-request kernel (on g16b) -> dev/modal/_slide/<kind>.py.
MODE | 16 (q_len 8, per-request): 512 threads = two 256-thread streams, each a full copy of the per-request segment
(own Smem, own named barriers) on one half of the split's tiles; stream 1 hands its (offset, row sum, o) state to
stream 0 through shared memory, stream 0 combines and runs the single global merge for the whole split."""
import pathlib, sys
H = pathlib.Path(__file__).resolve().parent
ROOT = H.parents[1]
SRC = (H / 'g16b.py').read_text()
OUT = ROOT / 'dev/modal/_slide'

def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)

def dual(s):
    s = rep(s, '''template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory");
  else __syncthreads();
}''', '''template <bool WS, int NT, bool DU = false>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory");
  else if constexpr (DU) asm volatile("bar.sync %0, 256;\\n" ::"r"(7 + (int)(threadIdx.x >> 8)) : "memory");   // this stream
  else __syncthreads();
}''')
    s = rep(s, '  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n',
            '  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n  int dual;             // q_len 8 per-request: the two-stream (512-thread) build\n')
    a = s.index('__device__ __forceinline__ bool segment(')
    z = s.index('// Flat grids whose CTAs are all resident')
    seg = s[a:z]
    seg = seg.replace('csync<WS, NT>()', 'csync<WS, NT, DUAL>()')
    seg = rep(seg, '  constexpr bool ONE = !FLAT && (MODE & 4) != 0;\n',
              '  constexpr bool ONE = !FLAT && (MODE & 4) != 0;\n'
              '  constexpr bool DUAL = !FLAT && L == 8 && (MODE & 16) != 0;   // stream = threadIdx.x >> 8\n')
    seg = rep(seg, 'const int tid = threadIdx.x, lane = tid & 31,',
              'const int tid = DUAL ? (int)(threadIdx.x & 255u) : (int)threadIdx.x, lane = tid & 31,')
    seg = rep(seg, '  const int wgi = hk;          // named barrier of this warpgroup',
              '  const int wgi = hk + (DUAL ? 2 * (int)(threadIdx.x >> 8) : 0);   // named barrier of this warpgroup (3..6)')
    seg = rep(seg, "      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)\n      __syncthreads();",
              "      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)\n      if constexpr (DUAL) csync<WS, NT, DUAL>(); else __syncthreads();")
    seg = rep(seg, "   __syncthreads();                   // shared memory is reused by the next segment",
              "   if constexpr (DUAL) csync<WS, NT, DUAL>(); else __syncthreads();   // shared memory is reused by the next segment")
    seg = seg.replace('const int mine = end - start;', 'const int mine = (DUAL ? S.fce : end) - start;   // (dual: the whole split)')
    assert seg.count('(DUAL ? S.fce : end)') == 2
    seg = rep(seg, '''  bool do_red = true;
''', '''  bool do_red = true;
  if constexpr (DUAL) {
    // stream 1 -> stream 0 through stream 1's (drained) stage memory: same local thread = same rows and dims
    const bool s1 = threadIdx.x >= 256;
    float* xs = reinterpret_cast<float*>(s1 ? S.k : (&S)[1].k) + tid * 36;
    if (s1) {
      cp_wait<0>();
      csync<WS, NT, DUAL>();
      xs[0] = m_r[0]; xs[1] = m_r[1]; xs[2] = l_r[0]; xs[3] = l_r[1];
      const float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) xs[4 + e] = od[e];
    }
    bar_sync_n(9, 512);
    if (s1) return true;
    float f0[2], f1[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float m1 = xs[r], l1 = xs[2 + r];
      const float mm = fmaxf(m_r[r], m1);
      f0[r] = (m_r[r] == -INFINITY) ? 0.f : ex2(m_r[r] - mm);
      f1[r] = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[r] = mm;
      l_r[r] = l_r[r] * f0[r] + l1 * f1[r];
    }
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) od[e] = od[e] * f0[(e >> 1) & 1] + xs[4 + e] * f1[(e >> 1) & 1];
  }
''')
    s = s[:a] + seg + s[z:]
    # kernel: 512 threads, two Smem copies, the split's tiles halved between the streams
    s = rep(s, '''template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(''',
            '''template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX * ((MODE & 16) ? 2 : 1), (MODE & 16) ? 1 : Cfg<L, FLAT>::MINB) attn_kernel(''')
    s = rep(s, '''  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));''',
            '''  Smem<L, FLAT>* Sp0 = reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  constexpr bool DUALK = !FLAT && L == 8 && (MODE & 16) != 0;
  Smem<L, FLAT>& S = (DUALK && threadIdx.x >= 256) ? Sp0[1] : Sp0[0];''')
    s = rep(s, '''    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys,''',
            '''    if constexpr (DUALK) {   // stream 0: the first half of the split's tiles, stream 1 the rest (may be empty)
      const int ntl = (end - start + TILE - 1) / TILE;
      const int mid = min(end, start + ((ntl + 1) >> 1) * TILE);
      if (threadIdx.x == 0) S.fce = end;
      if (threadIdx.x >= 256) start = mid; else end = mid;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys,''')
    # host
    s = rep(s, '''  if (q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid''', '''  if constexpr (L == 8) {
    if (p.dual && !p.flat && !cl && !pdl && (long long)grid.x * grid.y * grid.z <= nsm) {
      static bool dinit = false;
      const int smd = 2 * (int)sizeof(Smem<L, false>) + 1024;
      if (!dinit) {
        cudaFuncSetAttribute(attn_kernel<L, false, 19>, cudaFuncAttributeMaxDynamicSharedMemorySize, smd);
        cudaFuncSetAttribute(attn_kernel<L, false, 17>, cudaFuncAttributeMaxDynamicSharedMemorySize, smd);
        dinit = true;
      }
      (q.coop ? attn_kernel<L, false, 19> : attn_kernel<L, false, 17>)<<<grid, 2 * Cfg<L, false>::NTX, smd, stream>>>(q);
      return;
    }
  }
  if (q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid''')
    s = rep(s, '  p.coop = (int)((tma >> 1) & 1);\n', '  p.coop = (int)((tma >> 1) & 1);\n  p.dual = (int)((tma >> 4) & 1);\n')
    s = rep(s, 'tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT)',
            'tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT) | (int(plan.get("dual", 0)) << 4)')
    return s

TUN = '''    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),'''
def build(kind):
    s = SRC
    if kind == 'g16b':
        return rep(s, 'name=f"okc_attn_vfg16bs{tag}_90a"', 'name=f"okc_attn_slg16b{tag}_90a"')
    if kind.startswith('p'):
        sys.path.insert(0, str(H))
        import make_slide as MS
        s = MS.stamps(s)
        s = rep(s, '#define SPIN_CHECK', '#define STAMP1(k) do { if (p.dbg && threadIdx.x == 256) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); \\\n  p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k)] = _t; } } while (0)\n#define SPIN_CHECK')
    s = dual(s)
    if kind.startswith('p'):
        s = rep(s, "      if (i == 0) STAMP(3);\n      if (i == ntiles - 1) STAMP(4);\n", "      if (i == 0) STAMP(3);\n      if (i == ntiles - 1) STAMP(4);\n      if (i == 0) STAMP1(10);\n      if (i == ntiles - 1) STAMP1(11);\n", 2)
        s = rep(s, "      cp_wait<0>();\n      csync<WS, NT, DUAL>();\n      xs[0] = m_r[0];", "      STAMP1(12);\n      cp_wait<0>();\n      csync<WS, NT, DUAL>();\n      xs[0] = m_r[0];")
        s = rep(s, "    bar_sync_n(9, 512);\n", "    bar_sync_n(9, 512);\n    STAMP(9);\n")
        s = rep(s, "      if (tidL < 8) cp_async16(", "      STAMP1(13);\n      if (tidL < 8) cp_async16(") if s.count("      if (tidL < 8) cp_async16(") == 1 else s
        kind = kind[1:]
    if kind == 'w4d':     # w4a with 2 stages per stream (one tile in flight per stream: the SM keeps 2, as the 256-thread build)
        s = rep(s, '  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n  // flat q_len 8',
                '  constexpr int STAGES = DUAL ? 2 : Cfg<L, FLAT>::STAGES;\n  // flat q_len 8')
        kind = 'w4a'; tag = 'w4d'
    else:
        tag = kind
    if kind == 'w4a':     # v16s 8 splits (5 tiles: 3 + 2), v32s 4 splits (9 tiles: 5 + 4)
        s = rep(s, TUN, '''    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1, dual=1),
    (16, 8, True, 1): dict(n_splits=8, nc=1, coop=1, dual=1),''')
    elif kind == 'w4b':   # v16s 4 splits (9 tiles: 5 + 4)
        s = rep(s, TUN, '''    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1, dual=1),
    (16, 8, True, 1): dict(n_splits=4, nc=1, coop=1, dual=1),''')
    elif kind == 'w4c':   # v32s 3 splits (11 tiles: 6 + 5, 96 CTAs), v16s 6 splits (6 tiles: 3 + 3)
        s = rep(s, TUN, '''    (32, 8, True, 1): dict(n_splits=3, nc=1, coop=1, dual=1),
    (16, 8, True, 1): dict(n_splits=6, nc=1, coop=1, dual=1),''')
    else:
        raise SystemExit(kind)
    return rep(s, 'name=f"okc_attn_vfg16bs{tag}_90a"', f'name=f"okc_attn_sl{tag}{{tag}}_90a"')

if __name__ == '__main__':
    for k in sys.argv[1:]:
        (OUT / f'{k}.py').write_text(build(k)); print(k)
