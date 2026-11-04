"""pb ablations (no parent prepare): time each config with data paths switched off (dbg bits:
1 no decode/wgmma, 2 no x TMA, 4 no weight copy, 8 no split-K handoff wait)."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
sp = importlib.util.spec_from_file_location('candidate', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
S._pb_ext()
CFGS = json.loads(os.environ.get('CFGS', '[]'))
DBGS = json.loads(os.environ.get('DBGS', '[0, 1, 2, 4, 3, 5, 7, 8]'))
pbst = {'packs': {}, 'streams': {}}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    out = fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); torch.cuda.synchronize(); return g, out
def tm(g):
    flush.add_(1.0); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
zz = torch.zeros(1, device='cuda')
g0, _ = cap(lambda: zz.add_(1.0))
fixed = statistics.median([tm(g0) for _ in range(40)])
print('FIXED %.2f' % fixed, flush=True)
for layer, m, cfg in CFGS:
    cell = S.pb_cell(pbst, w, layer, *cfg)
    print('ATTR', layer, m, cfg, 'regs', S.pb_occ(cell, m, 2), 'maxthr', S.pb_occ(cell, m, 3), 'static', S.pb_occ(cell, m, 4),
          'resident', S.pb_occ(cell, m), 'ctas', S.pb_ctas(cell, m), flush=True)
    if S.pb_ctas(cell, m) > S.pb_occ(cell, m):
        continue
    d = R.generate_input(m, layer, 61)
    res = {}
    for dbg in DBGS:
        c2 = dict(cell); c2['dbg'] = dbg
        g, o = cap(lambda: S.pb_run(d['x'], c2))
        if dbg == 0:
            res['ok'] = R.check_implementation(d, o, w)[0]
        res[dbg] = round(statistics.median([tm(g) for _ in range(25)]) - fixed, 2)
    print('ABL', layer, m, cfg, json.dumps(res), flush=True)
print('DONE', flush=True)
