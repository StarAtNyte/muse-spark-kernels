"""Per-CTA OKC_DEBUG globaltimer timeline of the bs1 TINY and bs8 builds, TFIN=0 (control) vs TFIN=1, cold L2.
Driver-side patches only. Stamps (us from the earliest CTA start): start, routed, A_done, B_done (compute warps),
joined (after WS_JOIN), ready (barrier passed / token counter seen), final (kernel end of the final-item CTAs)."""
import importlib.util, json, sys, statistics
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
def dbg_patch(s):
    s = s.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif', 1)
    E = 'p.dbg[gridDim.x * 8 + blockIdx.x * 32 + %d]'
    a = '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
    s = s.replace(a, a + '  if (OKC_DEBUG && tid == 0 && p.dbg) ' + E % 31 + ' = gtimer();\n')
    a = '    for (int n = 0; ld_acquire(c) < nl * RB_B && n < 20000000; ++n) { if (OKC_TFW) __nanosleep(OKC_TFW); }\n'
    s = s.replace(a, a + '    if (OKC_DEBUG && p.dbg) ' + E % 30 + ' = gtimer();\n')
    a = '    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n'
    s = s.replace(a, a + '    if (OKC_DEBUG && p.dbg) ' + E % 30 + ' = gtimer();\n')
    return s
for k in list(S.ALT_SRCS):
    S.ALT_SRCS[k] = dbg_patch(S.ALT_SRCS[k])
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
G = st['grid']
flush = torch.empty(512 * 1024 * 1024 // 4, device='cuda')
VARS = [(0, ()), (1, ()), (1, ('-DOKC_TFW=64',)), (1, ('-DOKC_TFR=1',)), (1, ('-DOKC_TFW=64', '-DOKC_TFR=1'))]
def defs_for(bs, tfin, extra=()):
    d = tuple(st['ps'][bs]) if bs in st.get('ps', {}) else tuple(S.TINY_DEFINES if bs <= 4 else S.TINY2_DEFINES) + st['sfxs']
    d = tuple(x.replace('-DOKC_TFIN=1', f'-DOKC_TFIN={tfin}') for x in d)
    return d + tuple(extra) + ('-DOKC_DEBUG=1',)
for bs, seed in ((1, 7919), (8, 5041622730710591082)):
    d = R.generate_input(bs, 'ep8', seed)
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    for tfin, extra in VARS + VARS:
        ext = S._ext(defs_for(bs, tfin, extra))
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device='cuda')
        dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device='cuda')
        sc = list(st['scratch'])[:3] + [dbg]
        acc = {}
        for rep in range(12):
            flush.sum(); dbg.zero_(); torch.cuda.synchronize()
            ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
            if rep < 4: continue
            tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            cols = {'start': tt[:, 0], 'routed': tt[:, 1], 'A_done': tt[:, 2], 'B_done': tt[:, 3], 'joined': ex[:, 31], 'ready': ex[:, 30], 'final': tt[:, 5]}
            for n, col in cols.items():
                col = col[col > 0]
                if col.numel():
                    c = ((col - t00) / 1000).sort().values
                    acc.setdefault(n, []).append((float(c[0]), float(c[c.numel() // 2]), float(c[-1])))
        res = {n: [round(statistics.median(v[i] for v in vs), 2) for i in range(3)] for n, vs in acc.items()}
        print('TL', json.dumps(dict(bs=bs, seed=seed, tfin=tfin, extra=extra, **{k: v for k, v in res.items() if k in ('B_done', 'joined', 'ready', 'final')})), flush=True)
print('DONE', flush=True)
