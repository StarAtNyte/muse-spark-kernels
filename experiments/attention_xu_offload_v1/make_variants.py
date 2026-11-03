"""Generate diagnostic / candidate variants of the readable attention parent (q_len-8 flat P packing)."""
import re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attention_readable_fp32_path_v1/submission.py').read_text()
OUT = ROOT / 'dev/modal/_attnx'
OUT.mkdir(exist_ok=True)

HELPERS = r'''
// 2^x on the FMA/ALU pipes (no MUFU): round-to-nearest split x = n + f, |f| <= 1/2, degree-3 polynomial for 2^f,
// n added to the exponent field. x <= ~9 here; -inf / very negative inputs clamp to 2^-126 (E4M3 0 after packing).
__device__ __forceinline__ float ex2p(float x) {
  x = fmaxf(x, -126.f);
  const float t = x + 12582912.f;            // 1.5 * 2^23: the low mantissa bits of t hold round(x)
  const float f = x - (t - 12582912.f);      // [-0.5, 0.5]
  const float p = fmaf(fmaf(fmaf(0.0555041087f, f, 0.240226507f), f, 0.693147181f), f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
// FP16 probability path: the FP32 scores of a pair rounded to f16x2, scaled and offset in one fused f16x2 FMA
// (log2 units), one ex2.approx.f16x2 and one f16x2 -> e4m3x2 conversion per pair; QK, PV and all partials stay FP32.
__device__ __forceinline__ uint32_t h2pack(float lo, float hi) {
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r;
}
__device__ __forceinline__ uint32_t pexp2h(float lo, float hi, uint32_t sc2, uint32_t nd2) {
  uint32_t a = h2pack(lo, hi), y, e; uint16_t q;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(y) : "r"(a), "r"(sc2), "r"(nd2));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(e) : "r"(y));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(q) : "r"(e));
  return (uint32_t)q;
}
__device__ __forceinline__ uint32_t pexp4h(float x0, float x1, float x2, float x3, uint32_t sc2, uint32_t nd2) {
  return pexp2h(x0, x1, sc2, nd2) | (pexp2h(x2, x3, sc2, nd2) << 16);
}
__device__ __forceinline__ float exfake(float x) { return fmaf(x, 0.0625f, 0.5f); }
__device__ __forceinline__ uint32_t packfake(float a, float b, float c, float d) {
  return (__float_as_uint(a) >> 24) | ((__float_as_uint(b) >> 16) & 0xff00u) | ((__float_as_uint(c) >> 8) & 0xff0000u) | (__float_as_uint(d) & 0xff000000u);
}
'''
ANCHOR = '__device__ __forceinline__ uint32_t prmt('

def lines_pack2(src):
    i = src.index('            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0]')
    j = src.index('\n', src.index('            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10]', i))
    return i, j

def variant(tag, fn):
    s = SRC.replace('okc_attn_v298', f'okc_attn_x{tag}_')
    s = s.replace(ANCHOR, HELPERS + ANCHOR, 1)
    i, j = lines_pack2(s)
    block = fn(s[i:j])
    s = s[:i] + block + s[j:]
    (OUT / f'{tag}.py').write_text(s)

def sub_args(block, which):
    """which(line_idx, arg_idx) -> True: that ex2 uses ex2p."""
    out = []
    for li, line in enumerate(block.split('\n')):
        k = [0]
        def rep(m):
            a = k[0]; k[0] += 1
            return ('ex2p(' if which(li, a) else 'ex2(') + 'fmaf('
        out.append(re.sub(r'ex2\(fmaf\(', rep, line))
    return '\n'.join(out)

variant('base', lambda b: b)
variant('nomufu', lambda b: b.replace('ex2(fmaf(', 'exfake(fmaf('))
variant('nocvt', lambda b: b.replace('e4m3x4f(', 'packfake('))
variant('p25', lambda b: sub_args(b, lambda l, a: a == 3))
variant('p50', lambda b: sub_args(b, lambda l, a: a in (1, 3)))
variant('p75', lambda b: sub_args(b, lambda l, a: a != 0))
variant('p100', lambda b: sub_args(b, lambda l, a: True))
def h16(block):
    out = []
    for line in block.split('\n'):
        m = re.match(r'(\s*)(pa\[kk\]\[\d\]) = e4m3x4f\((.*)\);\s*$', line)
        assert m, line
        args = re.findall(r'(?:ex2|EXQ)\(fmaf\((sv\[\d+\]), ssc, -(d(?:lo|hi))\)', m.group(3))
        assert len(args) == 4 and len({a[1] for a in args}) == 1, line
        d = args[0][1]
        out.append(f"{m.group(1)}{m.group(2)} = pexp4h({', '.join(a[0] for a in args)}, sc2, nd2_{d[1:]});")
    return '\n'.join(out)

def h16_variant(tag, both):
    s = SRC.replace('okc_attn_v298', f'okc_attn_x{tag}_')
    s = s.replace(ANCHOR, HELPERS + ANCHOR, 1)
    pre = "const uint32_t sc2 = h2pack(ssc, ssc), nd2_lo = h2pack(-dlo, -dlo), nd2_hi = h2pack(-dhi, -dhi);\n"
    i, j = lines_pack2(s)
    # pack2 lambda: recompute the offsets' f16 pairs inside it (dlo/dhi may move before the repack)
    s = s[:i] + '            ' + pre + h16(s[i:j]) + s[j:]
    if both:
        a = s.index('        pa[kk][0] = e4m3x4f(EXQ(')
        b = s.index('\n', s.index('        pa[kk][3] = e4m3x4f(EXQ(', a))
        s = s[:a] + '        ' + pre + h16(s[a:b]) + s[b:]
    (OUT / f'{tag}.py').write_text(s)

h16_variant('h16', False)
h16_variant('h16all', True)
def plain(tag, a, b):
    s = SRC.replace('okc_attn_v298', f'okc_attn_x{tag}_')
    assert s.count(a) == 1, a
    (OUT / f'{tag}.py').write_text(s.replace(a, b))
plain('nopk2', 'constexpr bool PK2 = PCHK;', 'constexpr bool PK2 = false;')
print(sorted(p.name for p in OUT.iterdir()))
