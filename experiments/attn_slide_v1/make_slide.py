"""attn_slide_v1 variants of the board parent (g16b) -> dev/modal/_slide/<kind>.py.
probe: per-CTA globaltimer stamps (diagnostic).  sXY: non-flat STAGES X for q_len 1, Y for q_len 8."""
import pathlib, sys, re
H = pathlib.Path(__file__).resolve().parent
ROOT = H.parents[1]
SRC = (H / 'parent.py').read_text()
OUT = ROOT / 'dev/modal/_slide'

def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)

STAMP = r'''
#define STAMP(k) do { if (p.dbg && threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); \
  p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + (k)] = _t; } } while (0)
'''

def stamps(s):
    s = rep(s, '#define SPIN_CHECK', STAMP + '#define SPIN_CHECK')
    s = rep(s, '    const int seq_ld = __ldg(p.seq_lens + b);\n',
            '    STAMP(0);\n    if (p.dbg && threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));'
            ' p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 15] = sm; }\n'
            '    const int seq_ld = __ldg(p.seq_lens + b);\n')
    s = rep(s, '  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n', '  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  STAMP(1);\n')
    s = rep(s, '  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n',
            '  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n    STAMP(2);\n')
    s = rep(s, '      cp_wait<STAGES - 2>();\n      __syncthreads();\n      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n',
            '      cp_wait<STAGES - 2>();\n      __syncthreads();\n      if (i == 0) STAMP(3);\n      if (i == ntiles - 1) STAMP(4);\n      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);\n')
    s = rep(s, '      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)\n      __syncthreads();       // tile i landed; previous tile\'s PV done (V16 free)\n',
            '      cp_wait<STAGES - 2>();   // (timing probe only, wrong results)\n      __syncthreads();       // tile i landed; previous tile\'s PV done (V16 free)\n      if (i == 0) STAMP(3);\n      if (i == ntiles - 1) STAMP(4);\n')
    s = rep(s, '  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n',
            '  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n  STAMP(5);\n')
    s = rep(s, '    csync<WS, NT>();\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);',
            '    csync<WS, NT>();\n    STAMP(6);\n    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);')
    s = rep(s, '    S.flag = (old + mine == n_keys);\n  }\n  csync<WS, NT>();\n', '    S.flag = (old + mine == n_keys);\n  }\n  csync<WS, NT>();\n  STAMP(6);\n')
    s = rep(s, '  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;\n', '  STAMP(8);\n')
    s = rep(s, '     // consumers\' shared memory is reused by the next segment\n   __syncthreads();',
            '     // consumers\' shared memory is reused by the next segment\n   STAMP(7);\n   __syncthreads();')
    return s

def stages(s, s1, s8):
    return rep(s, 'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
               f'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : ((L == 1) ? {s1} : {s8});')

TUNED_OLD = """_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
    (16, 1, True, 1): dict(n_splits=8, coop=0, nc=1),   # H200 paired sweep: 1.029x on 16-request sliding decode
}"""
TUNED = {
    'tA': """_TUNED = {
    (32, 8, True, 1): dict(n_splits=8, cs=8, ncl=1, nc=1, coop=0),
    (16, 8, True, 1): dict(n_splits=8, cs=8, ncl=1, nc=1, coop=0),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
    (16, 1, True, 1): dict(n_splits=8, cs=8, ncl=1, nc=1, coop=0),
    (32, 1, True, 1): dict(n_splits=8, cs=8, ncl=1, nc=1, coop=0),
}""",
    'tB': """_TUNED = {
    (32, 8, True, 1): dict(n_splits=8, nc=1, coop=1),
    (16, 8, True, 1): dict(n_splits=16, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
    (16, 1, True, 1): dict(n_splits=8, coop=1, nc=1),
    (32, 1, True, 1): dict(n_splits=8, coop=1, nc=1),
}""",
}

def _flat(n16, n32, extra=''):
    return """_TUNED = {
    (32, 8, True, 1): dict(n_splits=%d, flat=1, nc=1),
    (16, 8, True, 1): dict(n_splits=%d, flat=1, nc=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
    (16, 1, True, 1): dict(n_splits=%d, flat=1, nc=1),
    (32, 1, True, 1): dict(n_splits=%d, flat=1, nc=1),%s
}""" % (n32, n16, n16, n32, extra)
TUNED['fA'] = _flat(264, 264)
TUNED['fB'] = _flat(132, 264)
TUNED['fC'] = _flat(132, 132, """
    (1, 1, True, 1): dict(n_splits=32, flat=1, nc=1),
    (1, 8, True, 1): dict(n_splits=33, flat=1, nc=1),""")


RED8_OLD = """      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {"""
RED8_NEW = """      const bool odd = tig & 1;
      if constexpr (BFP) {   // 8 consecutive dims per op (v4.bf16x2): half the L2 atomic operations of v2
        float w[4][4];
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const int t = 2 * q4;
          const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
          const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
          const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
          const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
          w[q4][0] = odd ? r0 : a0; w[q4][1] = odd ? r1 : a1; w[q4][2] = odd ? b0 : r0; w[q4][3] = odd ? b1 : r1;
        }
        // w[q4]: dims 8 * (2 q4 + odd) + 4 * (tig >> 1) + 0..3; pair q4 = 2 qq (kept by tig < 2) with 2 qq + 1 (tig >= 2)
        const bool hi = (tig >> 1) & 1;
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          float x[4], y[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float snd = hi ? w[2 * qq][e] : w[2 * qq + 1][e];
            y[e] = __shfl_xor_sync(0xffffffffu, snd, 2);
            x[e] = hi ? w[2 * qq + 1][e] : w[2 * qq][e];
          }
          // !hi: own (dims 0..3) then received (4..7) of T = 2 (2 qq) + odd; hi: received (0..3) then own (4..7) of T = 2 (2 qq + 1) + odd
          const int T = 2 * (2 * qq + (hi ? 1 : 0)) + (odd ? 1 : 0);
          const float* lo4 = hi ? y : x;
          const float* hi4 = hi ? x : y;
          const __nv_bfloat162 u0 = __floats2bfloat162_rn(lo4[0], lo4[1]), u1 = __floats2bfloat162_rn(lo4[2], lo4[3]);
          const __nv_bfloat162 u2 = __floats2bfloat162_rn(hi4[0], hi4[1]), u3 = __floats2bfloat162_rn(hi4[2], hi4[3]);
          asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(acc_bf(p, acc + 8 * T)),
                       "r"(bf2u(u0)), "r"(bf2u(u1)), "r"(bf2u(u2)), "r"(bf2u(u3)) : "memory");
        }
      } else
#pragma unroll
      for (int t = 0; t < 8; t += 2) {"""

def r8v4(s):
    return rep(s, RED8_OLD, RED8_NEW)

RED1_OLD = """#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,"""
RED1_NEW = """      if constexpr (BFP) {   // dims 16 tig + 8 c + 0..7 of this thread: one v4.bf16x2 op each
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
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,"""

def bf1v4(s):
    s = rep(s, 'constexpr bool BFP = (L == 8) && !FLAT;', 'constexpr bool BFP = !FLAT && (L == 8 || (MODE & 8) == 0);')
    return rep(s, RED1_OLD, RED1_NEW)


def bf1m1(s):
    s = rep(s, 'constexpr bool BFP = (L == 8) && !FLAT;', 'constexpr bool BFP = !FLAT && (L == 8 || MODE == 1);')
    return rep(s, RED1_OLD, RED1_NEW)

def fwarm(s):
    # one CTA runs the last-CTA finalize dry (no stores) while its first tiles load: the finalize's code is in L2
    # (not DRAM) when the real last CTA of each request reaches it (every replay starts from a cold L2)
    s = rep(s, 'template <int L, int NT, bool BF>\n__device__ __forceinline__ void fin_nc1(',
            'template <int L, int NT, bool BF>\n__device__ __noinline__ void fin_nc1(')
    s = rep(s, '  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n',
            '  if constexpr (!FLAT && (MODE & 3) == 1) {\n'
            '    if (p.nc == 1 && blockIdx.x == 0 && blockIdx.y == gridDim.y - 1) fin_nc1<L, NT, BFP>(p, b, pair, bp, 1.f, 1);\n'
            '  }\n'
            '  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n')
    return s


def bf1x(s):
    s = rep(s, 'constexpr bool BFP = (L == 8) && !FLAT;', 'constexpr bool BFP = !FLAT && (L == 8 || (MODE & 11) == 1);')
    return rep(s, RED1_OLD, RED1_NEW)

def d16n(s, n):
    return rep(s, "(16, 1, True, 1): dict(n_splits=8, coop=0, nc=1),", f"(16, 1, True, 1): dict(n_splits={n}, coop=0, nc=1),")

def tuned(s, t):
    return rep(s, TUNED_OLD, t)

def prefetch(s):
    # L2 prefetch of the block's q / new k / new v rows at kernel entry, before the seq_lens load returns
    return rep(s, '    const int seq_ld = __ldg(p.seq_lens + b);\n', '''    {
      constexpr int QL = 20;   // 128-byte lines per token: 16 q heads, 2 k heads, 2 v heads of this pair
      const int t = threadIdx.x;
      if (t < L * QL) {
        const int j = t / QL, c = t - j * QL;
        const __nv_bfloat16* row = p.qkv + (ix_t)(b * L + j) * p.ld_qkv;
        const __nv_bfloat16* a = c < 16 ? row + (16 * pair + c) * 64
                               : (c < 18 ? row + p.Hq * 64 + (2 * pair + c - 16) * 64
                                         : row + (p.Hq + p.Hkv) * 64 + (2 * pair + c - 18) * 64);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
      }
    }
    const int seq_ld = __ldg(p.seq_lens + b);
''')

def bf1(s):
    s = rep(s, 'constexpr bool BFP = (L == 8) && !FLAT;', 'constexpr bool BFP = !FLAT && (L == 8 || (MODE & 8) == 0);')
    s = rep(s, '          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f,', '          acc_red4<BFP>(p, acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f,')
    return s

def build(kind):
    s = SRC
    if kind == 'base':
        pass
    elif kind == 'probe':
        s = stamps(s)
    elif kind.startswith('probe_s'):   # probe_sXY
        s = stamps(stages(s, int(kind[7]), int(kind[8])))
    elif re.fullmatch(r's\d\d', kind):
        s = stages(s, int(kind[1]), int(kind[2]))
    elif kind in TUNED:
        s = tuned(s, TUNED[kind])
    elif kind == 'pf':
        s = prefetch(s)
    elif kind == 'bf1':
        s = bf1(s)
    elif kind == 'r8v4':
        s = r8v4(s)
    elif kind == 'bf1v4':
        s = bf1v4(s)
    elif kind == 'bf1m1':
        s = bf1m1(s)
    elif kind == 'fwarm':
        s = fwarm(s)
    elif kind == 'fwb':
        s = fwarm(bf1m1(s))
    elif kind == 'bf1x':
        s = bf1x(s)
    elif kind in ('bx16', 'bx32', 'bx33'):
        s = d16n(bf1x(s), int(kind[2:]))
    elif kind == 'both':
        s = bf1v4(r8v4(s))
    else:
        raise SystemExit(kind)
    return rep(s, 'name=f"okc_attn_vfg16bs{tag}_90a"', f'name=f"okc_attn_sl{kind}{{tag}}_90a"')

if __name__ == '__main__':
    for k in sys.argv[1:]:
        (OUT / f'{k}.py').write_text(build(k)); print(k)
