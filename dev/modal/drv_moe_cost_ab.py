"""WGS stage-cost table shape (flatten factor f: c' = mean + f (c - mean), both phases) vs scored, MoE board sizes
32/128/256; reference-checked; paired cold-L2 graph medians; also phase-A/B finish spreads via debug stamps."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
A = '#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207'
B = '#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174'
src = S.CUDA_SRC
assert src.count(A) == 1 and src.count(B) == 1
def shape(line, f):
    head = ' '.join(line.split(' ')[:2]); vals = [int(v) for v in line.split(' ', 2)[2].split(',')]
    m = sum(vals) / len(vals)
    return head + ' ' + ', '.join(str(max(1, int(round(m + f * (v - m))))) for v in vals)
COMBOS = [(fa, fb) for fa in (0.5, 0.65, 0.8, 1.0) for fb in (1.0, 1.25, 1.5) if (fa, fb) != (1.0, 1.0)]
NAME = lambda fa, fb: f'a{int(fa*100):03d}b{int(fb*100):03d}'
for fa, fb in COMBOS:
    S.ALT_SRCS[NAME(fa, fb)] = src.replace(A, shape(A, fa)).replace(B, shape(B, fb))
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
main = tuple(S._main_defines(sfx)) + sfx
exts = {'base': S._ext(main), **{NAME(fa, fb): S._ext(main + ('@' + NAME(fa, fb),)) for fa, fb in COMBOS}}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(ext, d):
    x = d['hidden_states'].contiguous(); ids = d['topk_ids'].to(torch.int32).contiguous(); tw = d['topk_weights'].float().contiguous()
    out = torch.empty_like(x)
    fn = lambda: ext.okc_run(st['w'], st['scratch'], x, ids, tw, out, 0, 32, st['grid'])
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g, out
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bs in (32, 128, 256):
    agg = {k: [] for k in exts}
    for seed in (419, 7919):
        d = R.generate_input(bs, 'ep8', seed)
        graphs = {}
        for k, e in exts.items():
            g, o = cap(e, d)
            ok, msg = R.check_implementation(d, o, w)
            if not ok: print('CHECK_FAIL', bs, k, msg[:80], flush=True); continue
            graphs[k] = g
        names = list(graphs); ts = {k: [] for k in names}
        for rep in range(16):
            for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
        for k, v in ts.items(): agg[k].append(statistics.median(v))
    base = statistics.geometric_mean(agg['base'])
    print('ROW', json.dumps(dict(bs=bs, base=round(base, 1), **{f'gain_{k}': round(base / statistics.geometric_mean(v), 4) for k, v in agg.items() if k != 'base' and len(v) == 3})), flush=True)
print('DONE', flush=True)
