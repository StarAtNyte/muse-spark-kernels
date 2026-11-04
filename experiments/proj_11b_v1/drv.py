"""proj_11b_v1 screen: 11-bit g11 paths vs the parent's dispatch, per cell; reference checks + paired cold-L2 graph replays."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
t0 = time.time()
sp = importlib.util.spec_from_file_location('candidate', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
print('PREP done %.0fs' % (time.time() - t0), flush=True)
REPS = int(os.environ.get('REPS', '60'))
if S._G11_EXT is None:
    print('NO EXT'); sys.exit(1)
packs = {}
for (layer, m), cids in S._G11_CELLS.items():
    W = w[layer]
    for cid in cids:
        pk = S._g11_getpack(packs, layer, W, cid)
        fails, msg = 0, ''
        try:
            for seed, dist in ((11, 'normal'), (12, 'outliers'), (13, 'normal')):
                d = R.generate_input(m, layer, seed, dist=dist)
                out = S._g11_run(d['x'], pk, cid, m, d['config']['n']); torch.cuda.synchronize()
                ok = R.check_implementation(d, out, w)
                if not ok[0]:
                    fails += 1; msg = str(ok)[:200]
            d = R.generate_input(m, layer, 777)
            n = d['config']['n']
            tg, tb = S._g11_tpair(lambda: S._g11_run(d['x'], pk, cid, m, n), lambda: S._g11_parent_kernel(d, st), reps=REPS)
            tn, _ = S._g11_tpair(lambda: S._g11_run(d['x'], pk, cid, m, n, 1), lambda: S._g11_run(d['x'], pk, cid, m, n), reps=REPS // 2)
            print('ROW', json.dumps({'layer': layer, 'm': m, 'cid': cid, 'cfg': S._G11_CFGS[cid], 'fails': fails, 'g11_us': round(tg, 3), 'noesc_us': round(tn, 3),
                                     'parent_us': round(tb, 3), 'ratio': round(tg / tb, 4), 'msg': msg}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print('ERR', layer, m, cid, repr(exc)[:400], flush=True)
# end-to-end: the 9 judge tests through the candidate's kernel_fn
tests = [(1, 'qkv_gate', 'normal'), (7, 'pooler', 'normal'), (33, 'o', 'normal'), (130, 'unpooler', 'normal'), (5, 'unpooler', 'normal'),
         (8, 'router', 'normal'), (9, 'router', 'outliers'), (512, 'qkv_gate', 'outliers'), (1000, 'unpooler', 'outliers')]
cells = [(m, l, 'normal') for (l, m) in S._G11_CELLS]
for i, (m, layer, dist) in enumerate(tests + cells):
    d = R.generate_input(m, layer, 4000 + i, dist=dist)
    out = S.kernel_fn(d, st); torch.cuda.synchronize()
    ok = R.check_implementation(d, out, w)
    print('E2E', m, layer, dist, ok[0], str(ok[1])[:120], flush=True)
print('ROUTES', sorted(st['g11']['routes']), flush=True)
print('DONE %.0fs' % (time.time() - t0), flush=True)
