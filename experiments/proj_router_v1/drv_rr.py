"""proj_router_v1 screen: the parent's router route (dispatcher) vs rr configs per scored m. Reference checks
(check_implementation on normal / outlier inputs, plus _rr_valid's FP64 probes) and paired, interleaved cold-L2 graph
replays in one process. CANDS env (json {m: [[WG,T,QS,CL,ROT,S],...]}) overrides the list."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
t0 = time.time()
sp = importlib.util.spec_from_file_location('candidate', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
DEFAULT = {
    128: [(1, 64, 6, 1, 0, 16), (1, 64, 6, 1, 1, 16), (1, 64, 6, 2, 0, 16), (1, 64, 6, 2, 1, 16), (1, 32, 8, 4, 1, 8),
          (1, 128, 4, 1, 1, 33)],
    256: [(1, 64, 6, 1, 0, 8), (1, 64, 6, 1, 1, 8), (1, 64, 6, 2, 1, 8), (1, 64, 6, 4, 1, 8), (1, 64, 6, 4, 0, 8),
          (1, 128, 6, 2, 1, 16), (2, 64, 4, 4, 1, 8)],
}
CANDS = {int(k): [tuple(c) for c in v] for k, v in json.loads(os.environ['CANDS']).items()} if os.environ.get('CANDS') else DEFAULT
REPS = int(os.environ.get('REPS', '50'))
cfgs = sorted({c for v in CANDS.values() for c in v})
S._RR_ROUTES = {}
st = S.prepare(w)
print('PREP parent done %.0fs' % (time.time() - t0), flush=True)
t1 = time.time(); rst = S._rr_state(w, cfgs); print('RR compile %.0fs' % (time.time() - t1), flush=True)
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
FIX = statistics.median([tm(g0) for _ in range(60)])
print('FIXED one-node graph %.2f us' % FIX, flush=True)
w32 = w['router'].float()
for m in sorted(CANDS):
    d = R.generate_input(m, 'router', 777)
    gb, _ = cap(lambda: S._rr_parent_kernel(d, st))
    for c in CANDS[m]:
        bad, msg = 0, ''
        try:
            for seed, dist in ((11, 'normal'), (12, 'outliers'), (13, 'normal')):
                dd = R.generate_input(m, 'router', seed, dist=dist)
                for _ in range(2):
                    out = S._rr_run(dd['x'], rst, c)
                torch.cuda.synchronize()
                ok = R.check_implementation(dd, out, w)
                if not ok[0]:
                    bad += 1; msg = str(ok)[:200]
                else:
                    msg = str(ok[1])[:90]
            if not S._rr_valid(rst, c, m, w32):
                bad += 1; msg += ' | rr_valid failed'
            gc, _ = cap(lambda: S._rr_run(d['x'], rst, c))
            tb, tc, rat = [], [], []
            for i in range(REPS):
                if i % 2:
                    b_ = tm(gb); c_ = tm(gc)
                else:
                    c_ = tm(gc); b_ = tm(gb)
                tb.append(b_); tc.append(c_); rat.append((c_ - FIX) / max(b_ - FIX, 0.05))
            mb, mc = statistics.median(tb), statistics.median(tc)
            print('ROW', json.dumps({'m': m, 'cfg': c, 'res': S._rr_run(d['x'], rst, c, 1), 'fails': bad, 'base_us': round(mb - FIX, 3), 'cand_us': round(mc - FIX, 3),
                                     'ratio': round(statistics.median(rat), 3), 'msg': msg}), flush=True)
            del gc
        except Exception as exc:  # noqa: BLE001
            print('ERR', m, c, repr(exc)[:300], flush=True)
    del gb
print('DONE %.0fs' % (time.time() - t0), flush=True)
