"""Selective BF16 split-KV partials on top of g16b (RULES.md organizer clarification 2026-11-03/04): only the
per-request q_len-8 kernels (L == 8 && !FLAT: verify sliding at every batch, B=1 verify full) red-add bf16 partials
(red.global.add.noftz.v2.bf16x2); q_len 1 and the flat q_len-8 kernel keep FP32 (measured: bf16 there is slower).
Writer and readers share the compile-time choice BF. Row sums (ws_l) and the m stats stay FP32."""
import pathlib, sys, re
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'dev/modal/_vfull'


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


def build(base, tag):
    s = (OUT / f'{base}.py').read_text()
    helpers = '''// ---- split-KV partials, FP32 or (BF) bf16 at the same element indices of Params::ws_acc; callers pass the element's
// float* "address" either way. BF = per-request q_len 8 kernels only. Row sums (ws_l) stay FP32.
__device__ __forceinline__ uint32_t bf2u(__nv_bfloat162 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __nv_bfloat16* acc_bf(const Params& p, const float* a) {
  return reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + (a - p.ws_acc);
}
template <bool BF>
__device__ __forceinline__ void acc_red4(const Params& p, float* a, float x, float y, float z, float w) {
  if constexpr (BF) {
    const __nv_bfloat162 u = __floats2bfloat162_rn(x, y), v = __floats2bfloat162_rn(z, w);
    asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};" ::"l"(acc_bf(p, a)), "r"(bf2u(u)), "r"(bf2u(v)) : "memory");
  } else {
    red_v4(a, x, y, z, w);
  }
}
template <bool BF>
__device__ __forceinline__ float4 acc_ld4(const Params& p, const float* a, int h) {
  if constexpr (BF) {
    const uint2 r = __ldcg(reinterpret_cast<const uint2*>(acc_bf(p, a) + 4 * h));
    return make_float4(__uint_as_float(r.x << 16), __uint_as_float(r.x & 0xffff0000u), __uint_as_float(r.y << 16),
                       __uint_as_float(r.y & 0xffff0000u));
  } else {
    return __ldcg(reinterpret_cast<const float4*>(a) + h);
  }
}
template <bool BF>
__device__ __forceinline__ void acc_zero4(const Params& p, float* a, int h) {
  if constexpr (BF) *reinterpret_cast<uint2*>(acc_bf(p, a) + 4 * h) = make_uint2(0u, 0u);
  else *(reinterpret_cast<float4*>(a) + h) = make_float4(0.f, 0.f, 0.f, 0.f);
}

'''
    s = rep(s, '// Cluster mode, several clusters per request', helpers + '// Cluster mode, several clusters per request')
    # cluster_multi: per-request kernels only
    s = rep(s, '      red_v4(acc, a.x, a.y, a.z, a.w);', '      acc_red4<L == 8>(p, acc, a.x, a.y, a.z, a.w);')
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4))',
            'acc_ld4<L == 8>(p, p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4, 0)')
    s = rep(s, '*reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);',
            'acc_zero4<L == 8>(p, p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4, 0);')
    # sum_copies / fin_nc1: a BF template parameter, every access through the helpers
    s = rep(s, 'template <int L>\n__device__ __forceinline__ void sum_copies(', 'template <int L, bool BF>\n__device__ __forceinline__ void sum_copies(')
    s = rep(s, 'template <int L, int NT>\n__device__ __forceinline__ void fin_nc1(', 'template <int L, int NT, bool BF>\n__device__ __forceinline__ void fin_nc1(')
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(acc) + 1)', 'acc_ld4<BF>(p, acc, 1)', 4)
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(acc))', 'acc_ld4<BF>(p, acc, 0)', 4)
    s = rep(s, '*(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);', 'acc_zero4<BF>(p, acc, 1);', 4)
    s = rep(s, '*reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);', 'acc_zero4<BF>(p, acc, 0);', 4)
    s = rep(s, '      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);', '      sum_copies<L, BFP>(p, bp, j, hh, part, act, y, lsum);')   # segment
    s = rep(s, '      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);', '      sum_copies<L, BFP>(p, bp, j, hh, part, true, y, lsum);')
    s = rep(s, '    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);', '    sum_copies<L, false>(p, bp, j, hh, part, act, y, lsum);')   # flat_finalize
    s = rep(s, '    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);', '    fin_nc1<L, NT, BFP>(p, b, pair, bp, vscale, 0);')
    # the q_len 8 writer in segment
    s = rep(s, '        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);',
            '        acc_red4<BFP>(p, acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);')
    assert 'float4*>(acc' not in s
    # BFP: constexpr in segment's scope (next to ONE)
    s = rep(s, '  constexpr bool ONE = !FLAT && (MODE & 4) != 0;\n',
            '  constexpr bool ONE = !FLAT && (MODE & 4) != 0;\n  constexpr bool BFP = (L == 8) && !FLAT;   // bf16 split-KV partials (per-request q_len 8 kernels)\n')
    s = re.sub(r'name=f"okc_attn_vf\w+?\{tag\}_90a"', f'name=f"okc_attn_vf{tag}{{tag}}_90a"', s)
    (OUT / f'{tag}.py').write_text(s)
    print(tag, s.count('<BFP>'), s.count('acc_ld4<'))


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
