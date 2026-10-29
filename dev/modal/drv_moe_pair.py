"""MoE board paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on all scored sizes plus small
correctness sizes; every output reference-checked; alternating cold-L2 graph replays; several seeds."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/kernels/muse_moe_w4a16/submission.py'); C = load('cand', '/w/run/sub.py')
w = R.generate_weights(R.WEIGHTS_SEED)
sb, sc = B.prepare(w), C.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for bs in (2, 3, 4, 5):  # correctness only
    for seed in (5, 6, 7):
        for dist in ('normal', 'skewed'):
            d = R.generate_input(bs, 'ep8', seed, dist=dist)
            ok, msg = R.check_implementation(d, C.kernel_fn(d, sc), w)
            if not ok: print('CHECK_FAIL', bs, seed, dist, msg, flush=True)
print('SMALL_CHECKS_DONE', flush=True)
for bs in (1, 8, 16, 32, 128, 256):
    rb, rc = [], []
    for seed in (419, 7919, 15241, 3, 77, 1001):
        d = R.generate_input(bs, 'ep8', seed)
        gb, ob = cap(B, sb, d); gc, oc = cap(C, sc, d)
        ok, msg = R.check_implementation(d, oc, w)
        if not ok: print('CHECK_FAIL', bs, seed, msg, flush=True)
        tb, tc = [], []
        for rep in range(16):
            if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
            else: tb.append(t1(gb)); tc.append(t1(gc))
        rb.append(statistics.median(tb)); rc.append(statistics.median(tc))
    gb_, gc_ = statistics.geometric_mean(rb), statistics.geometric_mean(rc)
    print('ROW', json.dumps(dict(bs=bs, base=round(gb_, 2), cand=round(gc_, 2), gain=round(gb_ / gc_, 4))), flush=True)
print('DONE', flush=True)
