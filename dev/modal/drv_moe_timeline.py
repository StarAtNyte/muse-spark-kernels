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
for bs in (1, 8, 16, 32):
    if bs <= S.TINY_MAX_M: defs = tuple(S.TINY_DEFINES)
    elif bs <= S.TINY2_MAX_M: defs = tuple(S.TINY2_DEFINES)
    elif bs <= S.MID_MAX_M: defs = tuple(S.MID_DEFINES)
    else: defs = tuple(S._main_defines(sfx))
    ext = S._ext(defs + sfx + ('-DOKC_DEBUG=1',))
    for seed in (419, 7919):
        d = R.generate_input(bs, 'ep8', seed)
        x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
        ids = idt.long(); distinct = int(ids[(ids >= 0) & (ids < 32)].unique().numel())
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(st['scratch'])[:3] + [dbg]
        rows = []
        for rep in range(6):
            flush.add_(1); dbg.zero_(); torch.cuda.synchronize()
            ext.okc_run(st['w'], sc, x, idt, tw, out, 0, 32, G); torch.cuda.synchronize()
            rows.append(dbg[:G * 8].view(G, 8).cpu().double())
        tt = rows[-1]
        t00 = tt[:, 0][tt[:, 0] > 0].min()
        res = {}
        for k, n in NAMES.items():
            col = tt[:, k][tt[:, k] > 0]
            if col.numel():
                c = ((col - t00) / 1000).sort().values
                res[n] = [round(float(c[0]), 1), round(float(c[c.numel() // 2]), 1), round(float(c[-1]), 1)]
        print('TL', json.dumps(dict(bs=bs, seed=seed, distinct=distinct, **res)), flush=True)
print('DONE', flush=True)
