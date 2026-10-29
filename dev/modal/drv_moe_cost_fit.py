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
rowsA, yA, rowsB, yB = [], [], [], []
for bs in (32, 64, 128, 256):
    for seed in (419, 7919, 15241, 3):
        d = R.generate_input(bs, 'ep8', seed)
        x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device='cuda')
        dbg = torch.zeros(G * 12 + 1100, dtype=torch.int64, device='cuda')
        sc = list(st['scratch'])[:3] + [dbg]
        TA, TB = [], []
        for rep in range(5):
            flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
            ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double().numpy()
            TA.append((tt[:, 2] - tt[:, 6]) / 1000); TB.append((tt[:, 3] - tt[:, 2]) / 1000)
        TA, TB = np.median(TA[1:], 0), np.median(TB[1:], 0)
        rng = dbg[G * 8:G * 12].view(G, 4).cpu().numpy().astype(int)
        nj = int(dbg[G * 12].item()); jC = dbg[G * 12 + 1:G * 12 + 1 + nj].cpu().numpy()
        for (lo, hi, sh, rows, ys, T) in ((0, 1, 5, rowsA, yA, TA), (2, 3, 4, rowsB, yB, TB)):
            for c in range(G):
                f = np.zeros(9)
                for gs in range(rng[c, lo], rng[c, hi]):
                    j = gs >> (4 + sh); f[min(8, (int(jC[j]) + 7) // 8)] += 1
                f[0] = 1.0
                rows.append(f); ys.append(T[c])
for name, rows, ys in (('A', rowsA, yA), ('B', rowsB, yB)):
    X, y = np.array(rows), np.array(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    per = coef[1:]
    scaled = [int(round(v / per[0] * 112)) if per[0] > 0 else 0 for v in per] if name == 'A' else [int(round(v / per[0] * 100)) if per[0] > 0 else 0 for v in per]
    print('FIT', json.dumps(dict(phase=name, const_us=round(float(coef[0]), 2), stage_us=[round(float(v), 4) for v in per],
          r2=round(float(r2), 3), table_like_current=scaled)), flush=True)
print('DONE', flush=True)
