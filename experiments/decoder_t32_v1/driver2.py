"""decoder_t32_v1 round 2: (a) critical-path timeline at T=16/32 (per-kernel end-to-end increments), (b) PCX pre-wait
depths, (c) side-stream L2 prefetch of the o (and pooler head) weight streams while qkv/attention run."""
import importlib.util, json, statistics, sys, time
from unittest.mock import MagicMock
import torch
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer'); sys.path.insert(0, '/w/run')
import reference as R
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
import pf_ext
w = R.generate_weights(R.WEIGHTS_SEED)
tp = time.time(); st = S.prepare(w); torch.cuda.synchronize(); log('PREPARED', round(time.time() - tp, 1), 's')
bst = st['bfp']
pf_ext.pf_ext()
KF = S.kernel_fn
PF = []   # list of (tensor-getter(T), max bytes, delay_ns)
side = torch.cuda.Stream()
def kf(d, s):
    if not PF:
        return KF(d, s)
    T = d['attn_input'].shape[0] if d['attn_input'].dim() == 2 else d['attn_input'].shape[0] * d['attn_input'].shape[1]
    main = torch.cuda.current_stream(); side.wait_stream(main)
    with torch.cuda.stream(side):
        for get, mb, dl in PF:
            t = get(T).view(-1).view(torch.uint8); t = t[:min(t.numel(), mb)] if mb else t
            pf_ext.pf_ext().pf_l2(t, t.numel() // 16 * 16, 16384, 8, 128, dl, 1)
    o = KF(d, s); main.wait_stream(side); return o
S.kernel_fn = kf
o_stream = lambda T: bst['sk2'][S.bfp__sk2_name('o', T)][1]
p_stream = lambda T: bst['sk2']['pooler'][1]
PCX0 = dict(S.PCX_CFG)
def apply(v):
    PF.clear(); S.PCX_CFG.clear(); S.PCX_CFG.update(PCX0)
    for k, val in v.items():
        if k == 'pf': PF.extend(val)
        else: S.PCX_CFG[k] = val
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, v):
    apply(v)
    try:
        o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): o = S.kernel_fn(d, st)
        g.replay(); torch.cuda.synchronize(); return g, o
    finally:
        apply({})
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
from torch.profiler import profile, ProfilerActivity
def timeline(g, n=6):
    rows = {}
    for _ in range(n):
        flush.add_(1); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
        ev = sorted([e for e in p.events() if e.device_type.name == 'CUDA' and 'elementwise' not in e.name],
                    key=lambda e: e.time_range.start)
        if not ev: continue
        t0 = ev[0].time_range.start; pend = t0
        for j, e in enumerate(ev):
            s0, e0 = e.time_range.start - t0, e.time_range.end - t0
            inc = max(0, e.time_range.end - pend); pend = max(pend, e.time_range.end)
            rows.setdefault(j, []).append((e.name[:44], s0, e0, inc))
    out = []
    for j in sorted(rows):
        r = rows[j]; out.append((r[0][0], round(statistics.median(x[1] for x in r), 2), round(statistics.median(x[2] for x in r), 2),
                                 round(statistics.median(x[3] for x in r), 2)))
    return out
role16 = lambda P: {(r, 16): (P, 0, 0) for r in ('qkv', 'o', 'pool', 'un')}
role32 = lambda P: {(r, 32): (P, 0, 0) for r in ('qkv', 'o', 'pool', 'un')}
V = {16: {'base': {}, 'P1': role16(1), 'P3': role16(3), 'P5': role16(5),
          'pfo': {'pf': [(o_stream, 0, 0)]}, 'pfo6': {'pf': [(o_stream, 0, 6000)]},
          'pfoP': {'pf': [(o_stream, 0, 6000), (p_stream, 16 << 20, 9000)]}},
     32: {'base': {}, 'P1': role32(1), 'P3': role32(3), 'P8': role32(8),
          'pfo': {'pf': [(o_stream, 0, 0)]}, 'pfo8': {'pf': [(o_stream, 0, 8000)]},
          'pfoP': {'pf': [(o_stream, 0, 8000), (p_stream, 16 << 20, 11000)]}}}
base = dict(ctx=80000, page=1, dist='soft', spread='agentic')
CASES = [dict(batch=16, layer='sliding', seed=9301, **base), dict(batch=32, layer='sliding', seed=9302, **base),
         dict(batch=16, layer='full', seed=9303, **base), dict(batch=32, layer='full', seed=9304, **base),
         dict(batch=32, layer='sliding', seed=9305, **base), dict(batch=16, layer='sliding', seed=9306, **base)]
REPS = 30
for i, c in enumerate(CASES):
    T = c['batch']; vs = V[T]
    ls = R.case_fields(c, w)['local_start']; d = R.generate_input(**c, local_start=ls)
    gs, chk = {}, {}
    for nm, v in vs.items():
        try:
            g, o = cap(d, v); okk, msg = R.check_implementation(d, R.collect(d, o), w)
            chk[nm] = bool(okk)
            if okk: gs[nm] = g
            else: print('FAILCHK', nm, str(msg)[:200])
        except Exception as e:
            chk[nm] = 'EXC ' + repr(e)[:200]
    names = list(gs); ts = {n: [] for n in names}
    for rep in range(REPS):
        for n in (names if rep % 2 == 0 else names[::-1]): ts[n].append(t1(gs[n]))
    med = {n: round(statistics.median(x), 2) for n, x in ts.items()}
    rat = {n: round(statistics.median(a / b for a, b in zip(ts['base'], ts[n])), 4) for n in names if 'base' in ts}
    print('ROW', json.dumps(dict(T=T, layer=c['layer'], med_us=med, base_over_var=rat, ok=chk)), flush=True)
    if i < 4:
        for n in ('base', 'pfo6' if T == 16 else 'pfo8'):
            if n in gs:
                for r in timeline(gs[n]): print('TL', T, c['layer'], n, json.dumps(r), flush=True)
    del gs; torch.cuda.empty_cache()
log('DONE')
