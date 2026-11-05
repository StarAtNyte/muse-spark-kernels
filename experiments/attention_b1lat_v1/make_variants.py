"""Variants of the readable attention parent for batch-1 latency work -> dev/modal/_b1lat/*.py."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attention_readable_fp32_path_v1/submission.py').read_text()
OUT = ROOT / 'dev/modal/_b1lat'
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
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);''')
    s = rep(s, '''      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)''', '''      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
      if (i == 0) STAMP(3);''')
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
    s = rep(s, '''    csync<WS, NT>();
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);''', '''    csync<WS, NT>();
    STAMP(6);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);''')
    s = rep(s, '''    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
''', '''    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  STAMP(6);
''')
    s = rep(s, '''  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;
''', '''  STAMP(8);
''')
    s = rep(s, '''     // consumers' shared memory is reused by the next segment
   __syncthreads();                   // shared memory is reused by the next segment''', '''     // consumers' shared memory is reused by the next segment
   STAMP(7);
   __syncthreads();                   // shared memory is reused by the next segment''')
    return s

def write(tag, s):
    s = rep(s, 'okc_attn_v298', f'okc_attn_b1{tag}_')
    (OUT / f'{tag}.py').write_text(s)

VARIANTS = {'base': lambda s: s, 'stamp': stamps}
if __name__ == '__main__':
    for t in (sys.argv[1:] or VARIANTS):
        write(t, VARIANTS[t](SRC))
        print('wrote', t)
