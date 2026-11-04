"""Candidates derived from parent.py. spin: SPIN_NS compile-time poll sleep (env B1_SPIN at build, screening only)."""
import pathlib, sys
H = pathlib.Path(__file__).resolve().parent
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a)); return s.replace(a, b)
def spin(s):
    s = rep(s, '__nanosleep(64);', 'SPIN_SLEEP;', 4)
    s = rep(s, '#define SPIN_CHECK', '#ifndef SPIN_NS\n#define SPIN_NS 64\n#endif\n#if SPIN_NS > 0\n#define SPIN_SLEEP __nanosleep(SPIN_NS)\n#else\n#define SPIN_SLEEP do {} while (0)\n#endif\n#define SPIN_CHECK')
    s = s.replace('name=f"okc_attn_v298', 'name=f"okc_attn_b1v2s{_SPIN}')
    s = rep(s, '''                           "--expt-relaxed-constexpr", *flags],''', '''                           "--expt-relaxed-constexpr", f"-DSPIN_NS={_SPIN}", *flags],''')
    s = rep(s, '_MOD = None\n', '_MOD = None\nimport os as _os\n_SPIN = int(_os.environ.get("B1_SPIN", "64"))\n')
    return s
WSPF_ANCHOR = """      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }
"""
WSPF = WSPF_ANCHOR + """  if (!FLAT && p.B == 1) {
    // warm L2 with this request's FP32 workspace (all copies) and ticket counters: the judge's cold-L2 replays leave
    // them in DRAM, so the merge's reds and the ticket atomic would each wait on a DRAM fill on the critical path
    const int nla = p.nc * L * 32;                        // 128-byte lines of ws_acc (16 rows x 64 floats per copy, token)
    const int nll = (p.nc * L * 16 * 4 + 127) >> 7;       // lines of ws_l
    const float* wa = p.ws_acc + (ix_t)bp * p.nc * L * 1024;
    const float* wl = p.ws_l + (ix_t)bp * p.nc * L * 16;
    for (int e = (int)blockIdx.x * NT + tidL; e < nla + nll; e += (int)gridDim.x * NT) {
      const float* a = e < nla ? wa + e * 32 : wl + (e - nla) * 32;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
    }
    if (tidL == NT - 1) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_cnt + bp));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_cnt + 8192 + bp));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_cnt + 12288 + bp));
    }
  }
"""
def wspf(s):
    s = rep(s, WSPF_ANCHOR, WSPF)
    return rep(s, 'name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_v298w{tag}_90a"')
BULK_HELPER = """__device__ __forceinline__ void bulk_red_f32(float* gdst, const float* ssrc, int bytes) {
  // TMA bulk reduction: gdst[i] += ssrc[i] (FP32 adds performed in L2), one instruction for the whole partial
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\\n"
               ::"l"(gdst), "r"(smem_u32(ssrc)), "r"(bytes) : "memory");
}
"""
def bulk(s):
    a = """__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}
"""
    s = rep(s, a, a + BULK_HELPER)
    s = rep(s, """  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB""", """  // one request (B == 1): the CTA's FP32 partial is staged in shared memory (as in cluster mode) and added to the
  // workspace by one TMA bulk reduction instead of 256 (q_len 1) / 2048 (q_len 8) per-thread red.global.add.v4
  const bool BULK = !FLAT && !M_CL && p.B == 1;
  if (M_CL || BULK) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB""")
    s = rep(s, """    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk, false);
    return true;
  }
  if (do_red) {""", """    if (BULK) {
      fence_proxy_async();   // this thread's shared-memory partial -> visible to the async (bulk copy) proxy
      csync<WS, NT>();
      if (tid == 0) {
        bulk_red_f32(p.ws_acc + (ix_t)(bp * p.nc + copy) * L * 1024, PB, L * 16 * 64 * 4);
        bulk_red_f32(p.ws_l + (bp * p.nc + copy) * L * 16, PL, L * 16 * 4);
        asm volatile("cp.async.bulk.commit_group;\\ncp.async.bulk.wait_group 0;\\n" ::: "memory");
        asm volatile("fence.proxy.async.global;\\n" ::: "memory");   // async-proxy writes before the ticket's release
      }
    } else {
      csync<WS, NT>();
      cluster_finish<L>(p, S, b, pair, n_keys, chunk, false);
      return true;
    }
  }
  if (do_red && !BULK) {""")
    return s.replace('name=f"okc_attn_v298', 'name=f"okc_attn_v298k')
if __name__ == '__main__':
    (H / 'cand_bulk.py').write_text(spin(bulk((H / 'parent.py').read_text())))
    (H / 'cand_wspf.py').write_text(spin(wspf((H / 'parent.py').read_text())))
    src = (H / 'parent.py').read_text()
    (H / 'cand_spin.py').write_text(spin(src)); print('ok')
