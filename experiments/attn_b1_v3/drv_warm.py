"""Cold vs warm-code cost: T(graph[A; B]) - T(graph[A]) is B with its code, TLB and workspace warm but its inputs
cold (A and B use different input buffers). Same for the parent. Variants: b1 module flag sets (env FLAGSETS)."""
import importlib.util, json, os, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/modal/_b1lat/base.py'); C = load('cand', '/w/run/sub.py')
C._MOD = B._module()
FS = json.loads(os.environ.get('FLAGSETS', '[[]]'))
for f in FS: C._b1_module(tuple(f))
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fns):
    for f in fns: f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
SPEC = {
  'd1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1),
  'v1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=8),
  'd1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=1),
}
PROF = os.environ.get('PROF') == '1'
for cn in os.environ.get('CASES', 'd1s,v1s').split(','):
    da = R.generate_input(**SPEC[cn], seed=11); db = R.generate_input(**SPEC[cn], seed=4242)
    impl = {'base': lambda d: B.kernel_fn(d)}
    for f in FS:
        impl['b1' + ''.join(f)] = (lambda d, f=tuple(f): C._b1_launch(d, flags=f))
    gs = {}
    for k, fn in impl.items():
        o = fn(db); torch.cuda.synchronize(); ok, msg = R.check_implementation(db, R.collect(db, o))
        gs[k + ':B'] = cap([lambda: fn(db)]); gs[k + ':A'] = cap([lambda: fn(da)]); gs[k + ':AB'] = cap([lambda: fn(da), lambda: fn(db)])
        print('CHK', cn, k, ok, str(msg)[:50], flush=True)
    names = list(gs); ts = {k: [] for k in names}
    for rep in range(24):
        r = rep % len(names); order = names[r:] + names[:r]
        if rep % 2: order = order[::-1]
        for k in order: ts[k].append(t1(gs[k]))
    med = {k: statistics.median(v) for k, v in ts.items()}
    res = {k: dict(cold=round(med[k + ':B'], 3), warm=round(med[k + ':AB'] - med[k + ':A'], 3)) for k in impl}
    print('ROW', json.dumps(dict(case=cn, res=res)), flush=True)
    if PROF:
        for f in FS:
            ff = tuple(f) + ('-DB1_PROF=1',)
            dbg = torch.zeros(256 * 16, dtype=torch.int64, device='cuda')
            for warm in (0, 1):
                dbg2 = torch.zeros(256 * 16, dtype=torch.int64, device="cuda"); fns = ([lambda: C._b1_launch(da, dbg=dbg2, flags=ff)] if warm else []) + [lambda: C._b1_launch(db, dbg=dbg, flags=ff)]
                gp = cap(fns); st = []
                for rep in range(12):
                    dbg.zero_(); t1(gp); a = dbg.view(-1, 16).cpu(); st.append(a[a[:, 0] > 0])
                rows = {}
                for k in range(11):
                    meds = []
                    for live in st:
                        v = live[:, k]; v = v[v > 0]
                        if len(v) == 0: continue
                        meds.append(((v - live[:, 0].min()).double() / 1000).median().item())
                    if meds: rows[k] = round(statistics.median(meds), 2)
                print('PROF', cn, ''.join(f), 'warm' if warm else 'cold', json.dumps(rows), flush=True)
print('DONE', flush=True)
