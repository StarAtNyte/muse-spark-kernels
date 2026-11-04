"""proj_big screen: parent's routes (base) vs pb configs on the 8 m=128/256 cells; paired cold-L2 graph replays,
reference checks (reference.check_implementation). CANDS env (json) overrides the config list."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
t_start = time.time()
sp = importlib.util.spec_from_file_location('candidate', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
S.PB_ROUTES = {}
t1 = time.time(); S._pb_ext(); print('PB compile %.0fs' % (time.time() - t1), flush=True)
if os.environ.get('COMPILE_ONLY'):
    sys.exit(0)
st = S.prepare(w)
print('PREP parent done %.0fs' % (time.time() - t_start), flush=True)
DEFAULT = {
    'pooler:128': [(2, 6, 1, 1, 4), (2, 3, 1, 2, 4), (2, 2, 1, 3, 4)],
    'unpooler:128': [(2, 6, 1, 1, 2), (2, 3, 1, 2, 2), (2, 2, 1, 3, 2)],
    'o:128': [(2, 6, 1, 1, 2), (2, 3, 1, 2, 2)],
    'qkv_gate:128': [(2, 6, 1, 1, 7), (2, 3, 1, 2, 7), (2, 2, 1, 3, 7)],
    'pooler:256': [(2, 6, 1, 1, 2), (2, 3, 1, 2, 2), (2, 2, 1, 3, 2)],
    'unpooler:256': [(2, 6, 1, 1, 1), (2, 3, 1, 2, 1), (2, 2, 1, 3, 1)],
    'o:256': [(2, 6, 1, 1, 1), (2, 3, 1, 2, 1)],
    'qkv_gate:256': [(2, 3, 1, 2, 3), (2, 2, 1, 3, 3)],
}
CANDS = json.loads(os.environ['CANDS']) if os.environ.get('CANDS') else DEFAULT
REPS = int(os.environ.get('REPS', '30'))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '61').split(',')]
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
print('FIXED one-node graph %.2f us' % fixed, flush=True)
summary = []
for key, cfgs in CANDS.items():
    layer, m = key.split(':'); m = int(m)
    cells = []
    for cfg in cfgs:
        try:
            cell = S.pb_cell(pbst, w, layer, *cfg)
            if cell is None:
                print('CELL', key, cfg, 'unavailable', flush=True); continue
            if S.pb_ctas(cell, m) > S.pb_occ(cell, m):
                print('CELL', key, cfg, 'SKIP not resident', S.pb_ctas(cell, m), S.pb_occ(cell, m), flush=True); continue
            ok, msg = S.pb_selftest(cell, w, m)
            print('CELL', key, cfg, 'selftest', ok, msg, 'ctas', S.pb_ctas(cell, m), 'resident', S.pb_occ(cell, m), 'esc_max', cell['emax'], flush=True)
            if S.pb_ctas(cell, m) > S.pb_occ(cell, m):
                print('CELL', key, cfg, 'SKIP not resident', flush=True); continue
            if ok:
                cells.append((cfg, cell))
        except Exception as e:
            print('CELL', key, cfg, 'ERROR', repr(e)[:300], flush=True)
    for seed in SEEDS:
        d = R.generate_input(m, layer, seed)
        gb, ob = cap(lambda: S._pb_prev_kernel(d, st))
        okb = R.check_implementation(d, ob, w)
        gs = [('base', gb)]
        res = {'base': okb[0]}
        for cfg, cell in cells:
            gc, oc = cap(lambda: S.pb_run(d['x'], cell))
            okc = R.check_implementation(d, oc, w)
            res[str(cfg)] = okc[0]
            if not okc[0]:
                print('CHECKFAIL', key, cfg, okc[1][:300], flush=True)
            gs.append((str(cfg), gc))
        times = {n: [] for n, _ in gs}
        for i in range(REPS):
            order = gs if i % 2 == 0 else gs[::-1]
            for n, g in order:
                times[n].append(tm(g))
        med = {n: statistics.median(v) for n, v in times.items()}
        b = med['base']
        row = {'cell': key, 'seed': seed, 'base_us': round(b, 2), 'base_net': round(b - fixed, 2), 'correct': res}
        for n in med:
            if n != 'base':
                row[n] = {'us': round(med[n], 2), 'net': round(med[n] - fixed, 2), 'gain_raw': round(b / med[n], 4),
                          'gain_net': round((b - fixed) / (med[n] - fixed), 4)}
        print('ROW', json.dumps(row), flush=True)
        summary.append(row)
print('DONE %.0fs' % (time.time() - t_start), flush=True)
