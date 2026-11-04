"""moe_stack_v1 paired screen: parent (/w/run/sub.py) vs candidate (/w/run/cand.py) on the 6 scored sizes, the 5 TASK
tests, small sizes, and a build-mixing sequence (FINPAR bs32 <-> MFIN bs128/33 <-> tiny). Alternating cold-L2 replays."""
import importlib.util, json, statistics, sys, time
import torch
T0 = time.time()
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/run/sub.py'); C = load('cand', '/w/run/cand.py')
w = R.generate_weights(R.WEIGHTS_SEED)
sb, sc = B.prepare(w), C.prepare(w)
print('PREPARED', round(time.time() - T0), flush=True)
fails = 0
def chk(tag, d, o):
    global fails
    ok, msg = R.check_implementation(d, o, w)
    if not ok: fails += 1; print('CHECK_FAIL', tag, msg, flush=True)
for bs, dist in ((4, 'normal'), (33, 'normal'), (257, 'normal'), (128, 'skewed'), (1, 'normal'), (2, 'skewed'), (5, 'normal'), (6, 'skewed'), (8, 'skewed'), (17, 'normal'), (24, 'skewed'), (32, 'skewed')):
    for seed in (5, 6):
        d = R.generate_input(bs, 'ep8', seed, dist=dist); chk((bs, dist, seed), d, C.kernel_fn(d, sc))
for k, bs in enumerate((32, 128, 32, 33, 32, 8, 32, 32, 1, 24, 256, 32)):  # build / parity mixing
    d = R.generate_input(bs, 'ep8', 900 + k); chk(('mix', k, bs), d, C.kernel_fn(d, sc))
print('CHECKS_DONE fails', fails, round(time.time() - T0), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
gains = []
for bs in (1, 8, 16, 32, 128, 256):
    rb, rc, rr = [], [], []
    for seed in (419, 7919, 15241, 3, 77, 1001):
        d = R.generate_input(bs, 'ep8', seed)
        gb, ob = cap(B, sb, d); gc, oc = cap(C, sc, d)
        chk(('timed', bs, seed), d, oc)
        tb, tc = [], []
        for rep in range(20):
            if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
            else: tb.append(t1(gb)); tc.append(t1(gc))
        rb.append(statistics.median(tb)); rc.append(statistics.median(tc))
        rr.append(statistics.median([x / y for x, y in zip(tb, tc)]))
        del gb, gc
    gb_, gc_ = statistics.geometric_mean(rb), statistics.geometric_mean(rc)
    gains.append(gb_ / gc_)
    print('ROW', json.dumps(dict(bs=bs, base=round(gb_, 2), cand=round(gc_, 2), gain=round(gb_ / gc_, 4),
                                 paired=round(statistics.geometric_mean(rr), 4), seeds=[round(x, 3) for x in rr])), flush=True)
print('BOARD_GEOMEAN', round(statistics.geometric_mean(gains), 4), 'fails', fails, round(time.time() - T0), flush=True)
print('DONE', flush=True)
