"""g16b: the linear exp2 -> e4m3 pack of g16 extended to the non-producer q_len-8 loop (per-request kernels:
sliding verify at every batch, B=1 full verify). Same format (e4m3 P, ones-column row sums), offsets snapped
so the f16 bias is exact. Diagnostic dq1: the q_len-1 loop with the MUFU ex2 removed (wrong results)."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = (ROOT / 'experiments/attn_vfull_v1/submission_g16.py').read_text()
OUT = ROOT / 'dev/modal/_vfull'


def rep(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:100])
    return s.replace(a, b)


def build(kind):
    s = SRC
    if kind == 'g16':
        return s
    if kind in ('g16b',):
        s = rep(s, '      if constexpr (PCHK) { dlo = lin_snap(dlo); dhi = lin_snap(dhi); cr[0] = dlo; cr[1] = dhi; }\n',
                '      dlo = lin_snap(dlo); dhi = lin_snap(dhi); cr[0] = dlo; cr[1] = dhi;\n')
        s = rep(s, 'const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);',
                'const float nlo = lin_snap(fmaxf(mlo, cr[0])), nhi = lin_snap(fmaxf(mhi, cr[1]));')
        old = '''        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));'''
        new = '''        {   // linear exp2 straight to e4m3 bytes (as the flat q_len 8 pack)
          const __half2 blo = lin_bias(dlo), bhi = lin_bias(dhi);
          pa[kk][0] = lin4h(sv[0], sv[1], sv[4], sv[5], lin_a, blo);
          pa[kk][1] = lin4h(sv[2], sv[3], sv[6], sv[7], lin_a, bhi);
          pa[kk][2] = lin4h(sv[8], sv[9], sv[12], sv[13], lin_a, blo);
          pa[kk][3] = lin4h(sv[10], sv[11], sv[14], sv[15], lin_a, bhi);
        }'''
        s = rep(s, old, new)
    elif kind == 'dq1':
        s = rep(s, '          const float pv = ex2(s[nt][e] - m_new);', '          const float pv = s[nt][e] - m_new;   // (diagnostic)')
    else:
        raise SystemExit(kind)
    return rep(s, 'name=f"okc_attn_vfg16{tag}_90a"', f'name=f"okc_attn_vf{kind}{{tag}}_90a"')


if __name__ == '__main__':
    for k in sys.argv[1:]:
        (OUT / f'{k}.py').write_text(build(k)); print(k)
