"""Per-CTA phase timeline of the scored MoE builds (OKC_DEBUG=1 globaltimer stamps) on reference inputs, cold L2.
Prints, per stamp, min / median / max over CTAs relative to the earliest start."""
import importlib.util, json, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
S.CUDA_SRC = S.CUDA_SRC.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
dev = 'cuda'
G = st['grid']
flush = torch.zeros(256 * 1024 * 1024 // 4, device=dev)
NAMES = {0: 'start', 1: 'routed', 7: 'gathered', 6: 'joined', 2: 'A_done', 3: 'B_done', 4: 'barrier', 5: 'final'}
sfx = st.get('sfx', ())
import numpy as np
for bs in (32, 128, 256):
    defs = tuple(S._main_defines(sfx))
    ext = S._ext(defs + sfx + ('-DOKC_DEBUG=1',))
    d = R.generate_input(bs, 'ep8', 419)
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
    dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
    sc = list(st['scratch'])[:3] + [dbg]
    A, Bd = [], []
    for rep in range(8):
        flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
        ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
        tt = dbg[:G * 8].view(G, 8).cpu().double(); t00 = tt[:, 0].min()
        A.append(((tt[:, 2] - t00) / 1000).numpy()); Bd.append(((tt[:, 3] - t00) / 1000).numpy())
    A, Bd = np.array(A[2:]), np.array(Bd[2:])
    def corr(M):
        cs = [np.corrcoef(M[i], M[j])[0, 1] for i in range(len(M)) for j in range(i + 1, len(M))]
        return round(float(np.mean(cs)), 3)
    print('ROW', json.dumps(dict(bs=bs, A_spread=round(float(np.mean(A.max(1) - A.min(1))), 2), A_std=round(float(np.mean(A.std(1))), 2),
          A_run_corr=corr(A), B_spread=round(float(np.mean(Bd.max(1) - Bd.min(1))), 2), B_std=round(float(np.mean(Bd.std(1))), 2),
          B_run_corr=corr(Bd), A_mean_slowest_ctas=[int(i) for i in np.argsort(-A.mean(0))[:6]])), flush=True)
print('DONE', flush=True)
