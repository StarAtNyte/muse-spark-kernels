"""Does the WGS scale multiply (bf16 HMUL2, FP16/tensor 'shared' pipe) cost tensor-core issue? Ablation: decoded
weights fed to wgmma without the scale HMUL2s (outputs wrong by design). Paired cold-L2 graph medians."""
import importlib.util, json, statistics, sys, os
print('START', flush=True)
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
OLD = '''          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);'''
assert S.CUDA_SRC.count(OLD) == 1
NEW = '''          af[ci & 1][jj][0] = r0 ^ (sgp & p.zero); af[ci & 1][jj][1] = r1;
          af[ci & 1][jj][2] = r2; af[ci & 1][jj][3] = r3;'''
S.ALT_SRCS['nohmul'] = S.CUDA_SRC.replace(OLD, NEW)
DQ = """          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);"""
assert S.CUDA_SRC.count(DQ) == 1, S.CUDA_SRC.count(DQ)
S.ALT_SRCS['nodq'] = S.CUDA_SRC.replace(DQ, """          uint32_t r0 = q, r1 = q ^ 0x1u, r2 = q ^ 0x2u, r3 = q ^ 0x4u;""")
S.ALT_SRCS['nodq_nohmul'] = S.ALT_SRCS['nodq'].replace(OLD, NEW)
X = '#define OKC_WGSX 0'
assert S.CUDA_SRC.count(X) == 1
M0 = """          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));"""
assert S.CUDA_SRC.count(M0) == 1, S.CUDA_SRC.count(M0)
NOMMA = """          d[0][0] += __uint_as_float((af[ci & 1][jj][0] ^ af[ci & 1][jj][1] ^ af[ci & 1][jj][2] ^ af[ci & 1][jj][3]) & p.zero) + (float)(uint32_t)(xd & p.zero);"""
S.ALT_SRCS['x8'] = S.CUDA_SRC.replace(X, '#define OKC_WGSX 8')
S.ALT_SRCS['nomma_x8'] = S.ALT_SRCS['x8'].replace(M0, NOMMA)
S.ALT_SRCS['bare'] = S.ALT_SRCS['nodq_nohmul'].replace(X, '#define OKC_WGSX 8').replace(M0, NOMMA)
GATH = "  wgs_gather(p, S);\n"
FIN = "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n"
XB = "      const uint32_t bytes = (uint32_t)npad * 512u;\n"
XC = """      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
"""
for t in (GATH, FIN, XB, XC): assert S.ALT_SRCS['bare'].count(t) == 1, t[:40]
b = S.ALT_SRCS['bare']
S.ALT_SRCS['bare_nox'] = b.replace(XB, "      const uint32_t bytes = 0u;\n").replace(XC, "")
S.ALT_SRCS['bare_nogather'] = b.replace(GATH, "")
S.ALT_SRCS['bare_nofinal'] = b.replace(FIN, "  if (p.zero == 0u) return;\n" + FIN)
S.ALT_SRCS['bare_all'] = S.ALT_SRCS['bare_nox'].replace(GATH, "").replace(FIN, "  if (p.zero == 0u) return;\n" + FIN)

NS = 'constexpr int WGS_NS = 3;'
NX = '#define OKC_WGSNX 3'
assert S.ALT_SRCS['bare_all'].count(NS) == 1 and S.ALT_SRCS['bare_all'].count(NX) == 1
BI = 'constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;'
BI2 = 'constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;'
assert S.CUDA_SRC.count(BI) == 1
S.ALT_SRCS['bare_ns4nx1'] = S.ALT_SRCS['bare_all'].replace(NS, 'constexpr int WGS_NS = 4;').replace(NX, '#define OKC_WGSNX 1').replace(BI, BI2)
S.ALT_SRCS['real_ns4nx2'] = S.CUDA_SRC.replace(NS, 'constexpr int WGS_NS = 4;').replace(NX, '#define OKC_WGSNX 2').replace(BI, BI2)
S.ALT_SRCS['bare_ns3nx1'] = S.ALT_SRCS['bare_all'].replace(NX, '#define OKC_WGSNX 1')
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
main = tuple(S._main_defines(sfx)) + sfx
exts = {} if os.environ.get('CANDIDATE_ONLY') == '1' else {'base': S._ext(main)}
import os
for k in (os.environ['VAR'],):
    try: exts[k] = S._ext(main + ('@' + k,))
    except Exception as e: print('BUILD_FAIL', k, str(e)[-200:], flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(ext, d):
    x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty_like(x)
    fn = lambda: ext.okc_run(st['w'], st['scratch'], x, ids, tw, out, 0, 32, st['grid'])
    print('CAP warmup', flush=True); fn(); torch.cuda.synchronize(); print('CAP warmup done', flush=True); g = torch.cuda.CUDAGraph()
    print('CAP begin', flush=True)
    with torch.cuda.graph(g): fn()
    print('CAP replay', flush=True); g.replay(); torch.cuda.synchronize(); print('CAP replay done', flush=True); return g, out
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bs in (32, 256):
    for seed in (419,):
        d = R.generate_input(bs, 'ep8', seed)
        graphs = {}
        for k, e in exts.items():
            try:
                g, o = cap(e, d)
            except Exception as ex:
                print('RUN_FAIL', k, str(ex)[:120], flush=True); raise
            if k.startswith('real'):
                ok, msg = R.check_implementation(d, o, w)
                if not ok: print('CHECK_FAIL', k, bs, seed, msg[:100], flush=True); continue
            graphs[k] = g
        if os.environ.get('CANDIDATE_ONLY') == '1': print('CANDIDATE_CHECK_PASSED', bs, flush=True); continue
        names = list(graphs); ts = {k: [] for k in names}
        for rep in range(30):
            for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(bs=bs, seed=seed, base=round(med['base'], 1),
              **{f'gain_{k}': round(med['base'] / v, 4) for k, v in med.items() if k != 'base'})), flush=True)
print('DONE', flush=True)
