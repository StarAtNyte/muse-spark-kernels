"""proj_o_v1 screen: parent's o route (dispatcher) vs ok_proj configs, per scored m; reference checks and paired
(interleaved) cold-L2 graph replays in one process. CANDS env (json {m: [[T,WG,KS,ST],...]}) overrides the list."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
t0 = time.time()
sp = importlib.util.spec_from_file_location('candidate', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
DEFAULT = {
    1: [(8, 2, 2, 8), (8, 1, 2, 8), (8, 1, 1, 12)],
    8: [(8, 2, 2, 8), (8, 1, 2, 8), (8, 1, 1, 12)],
    16: [(16, 2, 2, 8), (16, 1, 2, 8), (16, 1, 1, 12)],
    32: [(32, 2, 2, 8), (32, 1, 2, 8), (32, 1, 1, 10)],
    128: [(128, 2, 2, 6), (128, 1, 2, 4), (128, 1, 1, 6)],
    256: [(256, 2, 2, 4), (256, 1, 1, 5), (128, 2, 1, 6)],
}
CANDS = {int(k): [tuple(c) for c in v] for k, v in json.loads(os.environ['CANDS']).items()} if os.environ.get('CANDS') else DEFAULT
REPS = int(os.environ.get('REPS', '40'))
cfgs = []
for m in sorted(CANDS):
    for c in CANDS[m]:
        if c not in cfgs:
            cfgs.append(c)
t1 = time.time(); ost = S._ok_state(w, cfgs); print('OK compile %.0fs' % (time.time() - t1), flush=True)
S._OK_CANDS = {}
st = S.prepare(w)
print('PREP parent done %.0fs' % (time.time() - t0), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); torch.cuda.synchronize(); return g, out


def tm(g):
    flush.add_(1.0); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000


zz = torch.zeros(1, device='cuda')
g0, _ = cap(lambda: zz.add_(1.0))
print('FIXED one-node graph %.2f us' % statistics.median([tm(g0) for _ in range(40)]), flush=True)
for m in sorted(CANDS):
    for c in CANDS[m]:
        idx = cfgs.index(c)
        bad, msg = 0, ''
        ost['flags'].zero_(); torch.cuda.synchronize()
        try:
            for seed, dist in ((11, 'normal'), (12, 'normal'), (13, 'outliers')):
                d = R.generate_input(m, 'o', seed, dist=dist)
                out = S._ok_run(d['x'], ost, idx); torch.cuda.synchronize()
                ok = R.check_implementation(d, out, w)
                if not ok[0]:
                    bad += 1; msg = str(ok)[:160]
                for _ in range(3):   # repeat launches: the flag self-reset must hold
                    out2 = S._ok_run(d['x'], ost, idx)
                torch.cuda.synchronize()
                if not torch.equal(out, out2):
                    bad += 1; msg = 'nondeterministic repeat'
            d = R.generate_input(m, 'o', 777)
            gb, _ = cap(lambda: S._ok_parent_kernel(d, st))
            gc, _ = cap(lambda: S._ok_run(d['x'], ost, idx))
            tb, tc = [], []
            for _ in range(REPS):
                tb.append(tm(gb)); tc.append(tm(gc))
            mb, mc = statistics.median(tb), statistics.median(tc)
            for dbg in ():
                gd, _ = cap(lambda: S._ok_run(d['x'], ost, idx, dbg))
                td = [tm(gd) for _ in range(REPS)]
                print('DBG', m, c, dbg, round(statistics.median(td), 3), flush=True); del gd
            print('ROW', json.dumps({'m': m, 'cfg': c, 'fails': bad, 'base_us': round(mb, 3), 'cand_us': round(mc, 3),
                                     'ratio': round(mc / mb, 3), 'msg': msg}), flush=True)
            del gb, gc
        except Exception as exc:  # noqa: BLE001
            print('ERR', m, c, repr(exc)[:300], flush=True)
print('DONE %.0fs' % (time.time() - t0), flush=True)
