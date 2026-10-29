"""MoE board: which compiled build per scored token count (TINY / TINY2 / MID / MAIN at bs 1, 8, 16, 32),
reference-checked, paired cold-L2 graph medians over several reference seeds."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
B = {'tiny': tuple(S.TINY_DEFINES) + sfx, 'tiny2': tuple(S.TINY2_DEFINES) + sfx, 'mid': tuple(S.MID_DEFINES) + sfx,
     'main': tuple(S._main_defines(sfx)) + sfx}
exts = {k: S._ext(v) for k, v in B.items()}
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
CAND = {1: ('tiny', 'tiny2', 'mid')}
JS = 220742048650789979
for bs, names in CAND.items():
    agg = {k: [] for k in names}
    for seed in (JS, JS + 1, JS + 2, JS + 3, JS + 4, JS + 5):
        d = R.generate_input(bs, 'ep8', seed)
        ids = d['topk_ids'].long(); print('EXPERTS', seed - JS, int(ids[(ids >= 0) & (ids < 32)].unique().numel()), flush=True)
        graphs = {}
        for k in names:
            try:
                g, o = cap(exts[k], d)
                ok, msg = R.check_implementation(d, o, w)
                if ok: graphs[k] = g
                else: print('CHECK_FAIL', bs, k, msg[:80], flush=True)
            except Exception as e:
                print('RUN_FAIL', bs, k, str(e)[:80], flush=True)
        ts = {k: [] for k in graphs}
        for rep in range(16):
            for k in (list(graphs) if rep % 2 == 0 else list(graphs)[::-1]): ts[k].append(t1(graphs[k]))
        for k, v in ts.items(): agg[k].append(statistics.median(v))
    base = statistics.geometric_mean(agg[names[0]])
    print('ROW', json.dumps(dict(bs=bs, scored=names[0], **{k: round(statistics.geometric_mean(v), 2) for k, v in agg.items() if v},
          **{f'gain_{k}': round(base / statistics.geometric_mean(v), 4) for k, v in agg.items() if v and k != names[0]})), flush=True)
print('DONE', flush=True)
