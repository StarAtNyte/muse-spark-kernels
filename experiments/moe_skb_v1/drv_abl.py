"""Upper bound of any A->B readiness scheme: base vs base with every phase-B wait on phase A removed
(skinny doneA spins, MID DF sentinel spin, WGS per-(job,stage) doneAs spin). Timing only (nowait output is wrong)."""
import importlib.util, json, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
T0 = time.time()
def load(name, path, patch=None):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    if patch: m.CUDA_SRC = patch(m.CUDA_SRC)
    return m
def nowait(s):
    a = 'while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64);'
    assert s.count(a) == 3; s = s.replace(a, ';')
    b = 'if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }'
    assert s.count(b) == 1; s = s.replace(b, '')
    c = 'if (!__any_sync(0xffffffffu, bad != 0)) break;'
    assert s.count(c) == 1; s = s.replace(c, 'break;')
    return s + '\n// nowait\n'
w = R.generate_weights(R.WEIGHTS_SEED)
B = load('base', '/w/kernels/muse_moe_w4a16/submission.py'); sb = B.prepare(w); print('prep base', round(time.time() - T0), flush=True)
N = load('nowait', '/w/kernels/muse_moe_w4a16/submission.py', nowait); sn = N.prepare(w); print('prep nowait', round(time.time() - T0), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
gains = {}
for bs in (1, 8, 16, 32, 128, 256):
    med = {'base': [], 'nowait': []}
    for seed in (419, 7919, 15241, 3, 77, 1001):
        d = R.generate_input(bs, 'ep8', seed)
        gb, o = cap(B, sb, d); ok, msg = R.check_implementation(d, o, w)
        if not ok: print('CHECK_FAIL base', bs, msg, flush=True)
        gn, _ = cap(N, sn, d)
        ts = {'base': [], 'nowait': []}
        for rep in range(20):
            for k, g in ((('base', gb), ('nowait', gn)) if rep % 2 == 0 else (('nowait', gn), ('base', gb))): ts[k].append(t1(g))
        for k in ts: med[k].append(statistics.median(ts[k]))
    gm = {k: statistics.geometric_mean(v) for k, v in med.items()}
    gains[bs] = gm['base'] / gm['nowait']
    print('ROW', json.dumps(dict(bs=bs, base=round(gm['base'], 2), nowait=round(gm['nowait'], 2), gain=round(gains[bs], 4))), flush=True)
print('GEOMEAN_GAIN %.4f' % statistics.geometric_mean(list(gains.values())), 'DONE t=%d' % (time.time() - T0), flush=True)
