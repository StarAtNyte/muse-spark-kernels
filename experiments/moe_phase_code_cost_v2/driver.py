"""Which MoE phase pays the cold-code cost? Per-CTA stamps (OKC_DEBUG=1), cold L2 vs warm code + cold data."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
_fix = lambda t: t.replace('#define OKC_DEBUG 0', '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
S.CUDA_SRC = _fix(S.CUDA_SRC)
for _k in list(getattr(S, 'ALT_SRCS', {})): S.ALT_SRCS[_k] = _fix(S.ALT_SRCS[_k])
w1 = R.generate_weights(R.WEIGHTS_SEED); w2 = R.generate_weights(R.WEIGHTS_SEED + 1)
s1 = S.prepare(w1); s2 = S.prepare(w2)
dev = 'cuda'; G = s1['grid']
flush = torch.zeros(256 * 1024 * 1024 // 4, device=dev)
NAMES = {0: 'start', 1: 'routed', 7: 'gathered', 6: 'joined', 2: 'A_done', 3: 'B_done', 4: 'barrier', 5: 'final'}
sfx = s1.get('sfx', ())
def run(ext, st, d, dbg):
    x = d['hidden_states']; idt = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    if idt.data_ptr() & 15: idt = idt.clone()
    out = torch.empty_like(x); sc = list(st['scratch'])[:3] + [dbg]
    ext.okc_run(st['w'], sc, x, idt, tw, out, st['lo'], st['cnt'], G)
def stamps(dbg):
    tt = dbg[:G * 8].view(G, 8).cpu().double()
    if not (tt[:, 0] > 0).any(): return {}
    t00 = tt[:, 0][tt[:, 0] > 0].min(); res = {}
    for k, n in NAMES.items():
        col = tt[:, k][tt[:, k] > 0]
        if col.numel():
            c = ((col - t00) / 1000).sort().values
            res[n] = (float(c[0]), float(c[c.numel() // 2]), float(c[-1]))
    return res
for bs in (16, 32, 8):
    if bs <= S.TINY_MAX_M: defs = tuple(S.TINY_DEFINES)
    elif bs <= S.TINY2_MAX_M: defs = tuple(S.TINY2_DEFINES)
    elif bs <= S.MID_MAX_M: defs = tuple(S.MID_DEFINES)
    else: defs = tuple(S._main_defines(sfx))
    ext = S._ext(defs + sfx + ('-DOKC_DEBUG=1',))
    A = R.generate_input(bs, 'ep8', 419); B = R.generate_input(bs, 'ep8', 900 + bs)
    dA = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev); dB = dA.clone()
    cold, warm = [], []
    for rep in range(8):
        flush.add_(1); dA.zero_(); torch.cuda.synchronize(); run(ext, s1, A, dA); torch.cuda.synchronize(); cold.append(stamps(dA))
        flush.add_(1); dB.zero_(); torch.cuda.synchronize(); run(ext, s2, B, dB)      # other weights / inputs: leaves the code warm
        dA.zero_(); run(ext, s1, A, dA); torch.cuda.synchronize(); warm.append(stamps(dA))
    cold = [r for r in cold if r]; warm = [r for r in warm if r]
    if len(cold) < 3 or len(warm) < 3:
        print('PHASES bs', bs, 'no stamps recorded'); continue
    def med(rows, n, i): return statistics.median(r[n][i] for r in rows[2:] if n in r)
    print('PHASES bs', bs, '(median CTA time since first start, us; min/med/max)')
    for n in NAMES.values():
        if n in cold[-1]:
            print(f'  {n:9s} cold {med(cold,n,0):6.1f} {med(cold,n,1):6.1f} {med(cold,n,2):6.1f}   warm {med(warm,n,0):6.1f} {med(warm,n,1):6.1f} {med(warm,n,2):6.1f}   diff(med) {med(cold,n,1)-med(warm,n,1):+5.1f}', flush=True)
print('DONE', flush=True)
