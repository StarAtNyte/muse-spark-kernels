"""Why are WGS phase tails systematic? Export each CTA's stage ranges and the jobs' token counts (debug alt source),
then correlate per-CTA phase-A / phase-B finish times with range features."""
import importlib.util, json, sys
import numpy as np, torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
src = S.CUDA_SRC.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
OLD = '''    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);'''
assert src.count(OLD) == 1
src = src.replace(OLD, '''    if (threadIdx.x == 0 && p.dbg) {
      unsigned long long* q = p.dbg + gridDim.x * 8 + blockIdx.x * 4;
      q[0] = S.wgr[0]; q[1] = S.wgr[1]; q[2] = S.wgr[2]; q[3] = S.wgr[3];
      if (blockIdx.x == 0) {
        unsigned long long* r = p.dbg + gridDim.x * 12;
        const int nj = S.njF + S.njS; r[0] = nj;
        for (int j = 0; j < nj && j < 1000; ++j) r[1 + j] = S.jC[j];
      }
    }
''' + OLD)
S.ALT_SRCS['feat'] = src
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
ext = S._ext(tuple(S._main_defines(sfx)) + sfx + ('@feat', '-DOKC_DEBUG=1'))
G = st['grid']
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
for bs in (128, 256):
    d = R.generate_input(bs, 'ep8', 419)
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty((bs, 4096), dtype=torch.bfloat16, device='cuda')
    dbg = torch.zeros(G * 12 + 1100, dtype=torch.int64, device='cuda')
    sc = list(st['scratch'])[:3] + [dbg]
    A, Bt = [], []
    for rep in range(6):
        flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
        ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        tt = dbg[:G * 8].view(G, 8).cpu().double().numpy(); t00 = tt[:, 0].min()
        A.append((tt[:, 2] - t00) / 1000); Bt.append((tt[:, 3] - tt[:, 2]) / 1000)
    A, Bt = np.mean(A[1:], 0), np.mean(Bt[1:], 0)
    rng = dbg[G * 8:G * 12].view(G, 4).cpu().numpy()
    nj = int(dbg[G * 12].item()); jC = dbg[G * 12 + 1:G * 12 + 1 + nj].cpu().numpy()
    feats = {}
    for name, (lo, hi, sh) in (('A', (0, 1, 5)), ('B', (2, 3, 4))):
        g0, g1 = rng[:, lo].astype(int), rng[:, hi].astype(int)
        ns = g1 - g0
        js = [set(range(a >> (4 + sh), ((b - 1) >> (4 + sh)) + 1)) for a, b in zip(g0, g1)]
        njobs = np.array([len(s) for s in js])
        split_start = (g0 & 15) != 0; split_end = (g1 & 15) != 0
        ntt = np.array([np.mean([(jC[j] + 7) // 8 for j in s]) if s else 0 for s in js])
        y = A if name == 'A' else Bt
        def c(v): return round(float(np.corrcoef(v.astype(float), y)[0, 1]), 3)
        feats[name] = dict(spread=round(float(y.max() - y.min()), 2), corr_nstages=c(ns), corr_njobs=c(njobs),
                           corr_split_start=c(split_start), corr_split_end=c(split_end), corr_mean_ntt=c(ntt),
                           corr_cta_index=c(np.arange(G)), slow_cta_njobs=int(njobs[np.argmax(y)]), mean_njobs=round(float(njobs.mean()), 2))
    print('ROW', json.dumps(dict(bs=bs, nj=nj, jC_head=jC[:12].tolist(), **feats)), flush=True)
print('DONE', flush=True)
