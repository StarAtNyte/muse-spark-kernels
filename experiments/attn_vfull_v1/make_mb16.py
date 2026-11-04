"""BF16 split-KV partials (organizer clarification 2026-11-03/04 in RULES.md: partials and the combine may be BF16 within
tolerance). The workspace keeps its element layout (row x 64 dims) but holds bf16; every writer red-adds bf16x2
(red.global.add.noftz.bf16x2), every reader / zeroer converts. Row sums (ws_l) and the m stats stay FP32.
usage: make_mb16.py <base tag in dev/modal/_vfull> <out tag> [vec]"""
import pathlib, sys, re
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'dev/modal/_vfull'


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


def build(base, tag, vec):
    s = (OUT / f'{base}.py').read_text()
    red = ('''  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};" ::"l"(b), "r"(bf2u(u)), "r"(bf2u(v)) : "memory");'''
           if vec else
           '''  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(b), "r"(bf2u(u)) : "memory");
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(b + 2), "r"(bf2u(v)) : "memory");''')
    helpers = '''// ---- BF16 split-KV partials: the workspace (Params::ws_acc) holds bf16 at the same element indices; callers pass the
// element's float* "address" as before (row sums ws_l stay FP32)
__device__ __forceinline__ uint32_t bf2u(__nv_bfloat162 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __nv_bfloat16* acc_bf(const Params& p, const float* a) {
  return reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + (a - p.ws_acc);
}
__device__ __forceinline__ void acc_red4(const Params& p, float* a, float x, float y, float z, float w) {
  __nv_bfloat16* b = acc_bf(p, a);
  const __nv_bfloat162 u = __floats2bfloat162_rn(x, y), v = __floats2bfloat162_rn(z, w);
''' + red + '''
}
__device__ __forceinline__ float4 acc_ld4(const Params& p, const float* a, int h) {
  const uint2 r = __ldcg(reinterpret_cast<const uint2*>(acc_bf(p, a) + 4 * h));
  return make_float4(__uint_as_float(r.x << 16), __uint_as_float(r.x & 0xffff0000u), __uint_as_float(r.y << 16),
                     __uint_as_float(r.y & 0xffff0000u));
}
__device__ __forceinline__ void acc_zero4(const Params& p, float* a, int h) {
  *reinterpret_cast<uint2*>(acc_bf(p, a) + 4 * h) = make_uint2(0u, 0u);
}

'''
    s = rep(s, '// Cluster mode, several clusters per request', helpers + '// Cluster mode, several clusters per request')
    s = rep(s, 'red_v4(acc', 'acc_red4(p, acc', 3)
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(acc) + 1)', 'acc_ld4(p, acc, 1)', 4)
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(acc))', 'acc_ld4(p, acc, 0)', 4)
    s = rep(s, '*(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);', 'acc_zero4(p, acc, 1);', 4)
    s = rep(s, '*reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);', 'acc_zero4(p, acc, 0);', 4)
    s = rep(s, '__ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4))',
            'acc_ld4(p, p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4, 0)')
    s = rep(s, '*reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);',
            'acc_zero4(p, p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4, 0);')
    assert 'float4*>(acc' not in s and 'red_v4(acc' not in s
    s = re.sub(r'name=f"okc_attn_vf\w+?\{tag\}_90a"', f'name=f"okc_attn_vf{tag}{{tag}}_90a"', s)
    (OUT / f'{tag}.py').write_text(s)
    print(tag, s.count('acc_ld4'), s.count('acc_zero4'), s.count('acc_red4'))


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], len(sys.argv) > 3)
