from pathlib import Path
s=Path('dev/ptx_campaign/attention_verify_vswizzle.py').read_text()
a=s.index('_WG_CUDA ='); b=s.index('_WG_MOD =',a)
x=s[a:b]
a0=x.index('  {\n    const uint32_t ad ='); b0=x.index('  proxy_fence();',a0)
x=x[:a0]+x[b0:]
x=x.replace('9216','8192').replace('18432','16384').replace('1152','1024')
x=x.replace('float o[36];','float o[32];\n  float ls_run[2] = {0.f, 0.f};').replace('i < 36','i < 32').replace('j < 9','j < 8')
x=x.replace('m_run[0] = n0; m_run[1] = n1;', 'm_run[0] = n0; m_run[1] = n1;\n      ls_run[0] *= c0; ls_run[1] *= c1;')
x=x.replace('      pa[j][0] = pack_h2(p0, p1);','      ls_run[0] += p0 + p1; ls_run[1] += p2 + p3;\n      pa[j][0] = pack_h2(p0, p1);')
x=x.replace('wg_pv72(o,', 'wg_pvt(o,').replace('(kk * 144)', '(kk * 128)')
x=x.replace('const float ls = o[32 + 2 * h];', 'float ls = ls_run[h];\n    ls += __shfl_xor_sync(0xffffffffu, ls, 1);\n    ls += __shfl_xor_sync(0xffffffffu, ls, 2);')
s=s[:a]+x+s[b:]
s=s.replace('load_inline("muse_wg5_vswizzle",','load_inline("muse_wg5_pv64",')
Path('dev/ptx_campaign/attention_verify_pv64.py').write_text(s)
