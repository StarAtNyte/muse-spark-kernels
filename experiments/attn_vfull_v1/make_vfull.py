"""Build q_len-8 flat (full verify) variants of the readable parent that replace MUFU ex2 + F2FP e4m3 conversion
of P with a bit-level linear exp2 (Mitchell): the e4m3 byte of 2^x is ~ round(8 (x + 7)), so P bytes come from
FMA-pipe arithmetic with a magic-number round. Only the flat producer-mode pack (PK2 path, Cfg<8, true>) changes.

Variants (dev/modal/_vfull/<tag>.py):
  base : the parent, unchanged
  g16  : scores rounded to f16 (cvt f16x2), HFMA2.SAT to t = n/126 in [0,1], HFMA2 t*126 + 1024 (integer round),
         one PRMT per 4 scores. Offsets snapped so the f16 bias is exact.
  l32  : FP32 FFMA.SAT + FFMA magic 2^23, 3 PRMT per 4 scores.
QK, PV and all partials stay FP32 (organizer: exp2 approximations on FP16-rounded scores with E4M3 P allowed).
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attention_readable_fp32_path_v1/submission.py').read_text()
OUT = ROOT / 'dev/modal/_vfull'
OUT.mkdir(exist_ok=True)


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


HELPERS = r'''
// ---- linear (Mitchell) exp2 straight to e4m3 bytes: byte n = round(8 (x + 7)) encodes 2^x exactly at integer x and
// linearly in between (the mean bias cancels: the row sums come from the same bytes via the PV ones column).
// t = sat((8 (x + 7)) / 126) clamps to bytes [0, 126] (0 .. 448, never the 0x7f NaN); x = s * ssc - d.
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// f16 path: four raw scores, row bias b (f16x2), shared slope a (f16x2) -> four e4m3 bytes (x0 low)
__device__ __forceinline__ uint32_t lin4h(float x0, float x1, float x2, float x3, __half2 a, __half2 b) {
  const __half2 c126 = __half2half2(__ushort_as_half((unsigned short)0x57E0u));   // 126
  const __half2 c1024 = __half2half2(__ushort_as_half((unsigned short)0x6400u));  // 1024: ulp 1
  const __half2 t01 = __hfma2_sat(__floats2half2_rn(x0, x1), a, b);
  const __half2 t23 = __hfma2_sat(__floats2half2_rn(x2, x3), a, b);
  const uint32_t h01 = h2u(__hfma2(t01, c126, c1024)), h23 = h2u(__hfma2(t23, c126, c1024));
  return prmt(h01, h23, 0x6420u);
}
// offset snapped so that the f16 bias (56 - 8 d) / 126 is exact: returns the offset the bytes really use
__device__ __forceinline__ float lin_snap(float d) {
  const float bh = __half2float(__float2half_rn((56.f - 8.f * d) * (1.f / 126.f)));
  return (56.f - 126.f * bh) * 0.125f;
}
__device__ __forceinline__ __half2 lin_bias(float d) { return __float2half2_rn((56.f - 8.f * d) * (1.f / 126.f)); }
// FP32 path
__device__ __forceinline__ uint32_t lin4f(float x0, float x1, float x2, float x3, float a, float b) {
  const float y0 = fmaf(__saturatef(fmaf(x0, a, b)), 126.f, 8388608.f);
  const float y1 = fmaf(__saturatef(fmaf(x1, a, b)), 126.f, 8388608.f);
  const float y2 = fmaf(__saturatef(fmaf(x2, a, b)), 126.f, 8388608.f);
  const float y3 = fmaf(__saturatef(fmaf(x3, a, b)), 126.f, 8388608.f);
  const uint32_t lo = prmt(__float_as_uint(y0), __float_as_uint(y1), 0x0040u);
  const uint32_t hi = prmt(__float_as_uint(y2), __float_as_uint(y3), 0x0040u);
  return prmt(lo, hi, 0x5410u);
}
'''
ANCHOR = '// four floats -> four e4m3 bytes (x0 in the low byte), saturating\n'

PACK_OLD = '''          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
          }'''
PACK_G16 = '''          const __half2 blo = lin_bias(dlo), bhi = lin_bias(dhi);
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = lin4h(sv[0], sv[1], sv[4], sv[5], lin_a, blo);
            pa[kk][1] = lin4h(sv[2], sv[3], sv[6], sv[7], lin_a, bhi);
            pa[kk][2] = lin4h(sv[8], sv[9], sv[12], sv[13], lin_a, blo);
            pa[kk][3] = lin4h(sv[10], sv[11], sv[14], sv[15], lin_a, bhi);
          }'''
PACK_L32 = '''          const float blo = (56.f - 8.f * dlo) * (1.f / 126.f), bhi = (56.f - 8.f * dhi) * (1.f / 126.f);
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = lin4f(sv[0], sv[1], sv[4], sv[5], lin_a, blo);
            pa[kk][1] = lin4f(sv[2], sv[3], sv[6], sv[7], lin_a, bhi);
            pa[kk][2] = lin4f(sv[8], sv[9], sv[12], sv[13], lin_a, blo);
            pa[kk][3] = lin4f(sv[10], sv[11], sv[14], sv[15], lin_a, bhi);
          }'''
SSC_OLD = '  const float ssc = sl2;   // raw score -> log2 units\n'
FIRST_OLD = '''      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
'''
RESC_OLD = '''            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();'''


TUNED_OLD = '    (16, 1, True, 1): dict(n_splits=8, coop=0, nc=1),'


COOP_OLD = '''        coop = int(plan.get("coop", int(L == 8 and
                                        n_splits * pairs <= 2 * _num_sms(device))))'''


def build(kind):
    if kind.endswith('_c0'):   # flat q_len 8: last-CTA-per-request finalize instead of the grid barrier
        k0 = kind[:-3]
        s = build(k0)
        s = rep(s, COOP_OLD, '        coop = int(plan.get("coop", 0))')
        if k0 == 'base':
            return rep(s, 'name=f"okc_attn_v298{tag}_90a"', 'name=f"okc_attn_vfbc0{tag}_90a"')
        return s.replace(f'okc_attn_vf{k0}', f'okc_attn_vf{k0}c0')
    if '_nc' in kind:   # flat full-layer verify plans with nc workspace copies (fewer same-address L2 atomics)
        k0, nc = kind.split('_nc')
        s = build(k0)
        s = rep(s, TUNED_OLD, TUNED_OLD + f'\n    (16, 8, False, 1): dict(nc={nc}),\n    (32, 8, False, 1): dict(nc={nc}),')
        return s.replace(f'okc_attn_vf{k0}', f'okc_attn_vf{k0}n{nc}') if k0 != 'base' else rep(s, 'name=f"okc_attn_v298{tag}_90a"', f'name=f"okc_attn_vfbn{nc}{{tag}}_90a"')
    s = SRC
    if kind == 'base':
        return s
    s = rep(s, ANCHOR, HELPERS + ANCHOR)
    if kind == 'g16':
        s = rep(s, PACK_OLD, PACK_G16)
        s = rep(s, SSC_OLD, SSC_OLD + '  const __half2 lin_a = __float2half2_rn(ssc * (8.f / 126.f));   // (flat q8 linear exp2 slope)\n')
        s = rep(s, FIRST_OLD, FIRST_OLD + '      if constexpr (PCHK) { dlo = lin_snap(dlo); dhi = lin_snap(dhi); cr[0] = dlo; cr[1] = dhi; }\n')
        s = rep(s, RESC_OLD, RESC_OLD.replace('const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);',
                                              'const float nlo = lin_snap(fmaxf(hlo, cr[0])), nhi = lin_snap(fmaxf(hhi, cr[1]));'))
    elif kind == 'l32':
        s = rep(s, PACK_OLD, PACK_L32)
        s = rep(s, SSC_OLD, SSC_OLD + '  const float lin_a = ssc * (8.f / 126.f);   // (flat q8 linear exp2 slope)\n')
    elif kind == 'dnop':   # diagnostic: P = constant (no score work at all; wrong results)
        s = rep(s, PACK_OLD, '''          for (int kk = 0; kk < 2; ++kk) {
            pa[kk][0] = 0x30303030u; pa[kk][1] = 0x30303030u; pa[kk][2] = 0x30303030u; pa[kk][3] = 0x30303030u;
            asm volatile("" : "+r"(pa[kk][0]), "+r"(pa[kk][1]), "+r"(pa[kk][2]), "+r"(pa[kk][3]));
          }''')
    elif kind == 'dcons':   # diagnostic: consumers only wait and release stages (producer / memory ceiling)
        s = rep(s, '''      if (patched || i == 0) csync<WS, NT>();
''', '''      if (patched || i == 0) csync<WS, NT>();
      mbar_arrive(&S.empty[st_c]);
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      continue;
''')
    elif kind == 'dnopv':   # diagnostic: no PV wgmma in the flat q8 loop
        s = rep(s, '''          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);''', '''          asm volatile("" :: "r"(pa[0][0]), "r"(pa[0][1]), "r"(pa[0][2]), "r"(pa[0][3]), "r"(pa[1][0]), "r"(pa[1][1]), "r"(pa[1][2]), "r"(pa[1][3]));''')
    elif kind in ('pd2', 'pd3', 'dconspd3'):   # page indices PDN tiles ahead (register ring) in the cp.async producer
        pdn = 3 if kind != 'pd2' else 2
        s = rep(s, '      if constexpr (!TMAPG_ON) load_pages(0, pg);\n',
                '      int pq1[PK], pq2[PK];\n      if constexpr (!TMAPG_ON) { load_pages(0, pg); load_pages(1, pq1); '
                + ('load_pages(2, pq2); ' if pdn == 3 else '') + '}\n')
        s = rep(s, '          load_pages(x + 1, pgn);   // in flight while this tile\'s copies issue\n',
                f'          load_pages(x + {pdn}, pgn);   // {pdn} tiles ahead\n')
        rot = ('          for (int k = 0; k < PK; ++k) { pg[k] = pq1[k]; pq1[k] = pq2[k]; pq2[k] = pgn[k]; }\n' if pdn == 3 else
               '          for (int k = 0; k < PK; ++k) { pg[k] = pq1[k]; pq1[k] = pgn[k]; }\n')
        s = rep(s, '          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];\n', rot)
        if kind == 'dconspd3':
            s = rep(s, '''      if (patched || i == 0) csync<WS, NT>();
''', '''      if (patched || i == 0) csync<WS, NT>();
      mbar_arrive(&S.empty[st_c]);
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      continue;
''')
    elif kind == 'tmapg':
        s = rep(s, '      constexpr bool TMAPG_ON = (L == 1);\n', '      constexpr bool TMAPG_ON = true;\n')
    elif kind == 'dmrg':   # diagnostic: no partial reds (merge traffic cost)
        s = rep(s, '  if (do_red) {   // (one branch', '  if (do_red && L == 1) {   // (one branch')
    elif kind == 'dconsmrg':   # diagnostic: no consumer work and no partial reds
        s = build('dcons')
        s = rep(s, '  if (do_red) {   // (one branch', '  if (do_red && L == 1) {   // (one branch')
        return s.replace('okc_attn_vfdcons', 'okc_attn_vfdconsmrg')
    elif kind == 'dst':   # diagnostic: the q8 partial reds as plain 16-byte stores (same addresses; wrong results)
        s = rep(s, '        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);',
                '''        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1)),
                     "f"(odd ? r0 : a0), "f"(odd ? r1 : a1), "f"(odd ? b0 : r0), "f"(odd ? b1 : r1) : "memory");''')
    elif kind in ('st3', 'st5'):   # q_len 8 flat ring depth
        s = rep(s, 'static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;', f'static constexpr int STAGES = F ? ((L == 1) ? 3 : {kind[2]}) : 3;')
    else:
        raise SystemExit(kind)
    s = rep(s, 'name=f"okc_attn_v298{tag}_90a"', f'name=f"okc_attn_vf{kind}{{tag}}_90a"')
    return s


if __name__ == '__main__':
    for k in (sys.argv[1:] or ['base', 'g16', 'l32']):
        t = build(k)
        if k != 'base':
            t = t.replace('name=f"okc_attn_v298{tag}_90a"', f'name=f"okc_attn_vf{k}{{tag}}_90a"')
        (OUT / f'{k}.py').write_text(t)
        print(k, len(t))
