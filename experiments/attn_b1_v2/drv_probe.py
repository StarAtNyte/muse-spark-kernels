"""Per-CTA globaltimer stamps of /w/run/sub.py (probe build) on batch-1 shapes, cold-L2 graph replays."""
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
sp = importlib.util.spec_from_file_location('probe', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
SPEC = {
  'd1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1),
  'd1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=1),
  'v1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=8),
}
def graph(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for cn in os.environ.get('CASES', 'd1s,v1s,d1f').split(','):
    d = R.generate_input(**SPEC[cn], seed=11)
    dbg = torch.zeros(1024 * 16, dtype=torch.int64, device='cuda')
    o = S._launch(d, dbg=dbg); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o))
    g0 = graph(lambda: S.kernel_fn(d)); gs = graph(lambda: S._launch(d, dbg=dbg))
    ev, st = [], []
    for rep in range(12):
        ev.append(t1(g0)); dbg.zero_(); t1(gs)
        a = dbg.view(-1, 16).cpu(); st.append(a[a[:, 0] > 0])
    rows = {}
    for k in (0, 1, 2, 3, 4, 5, 9, 12, 10, 11, 6, 8, 7):
        mins, meds, maxs = [], [], []
        for live in st:
            v = live[:, k]; v = v[v > 0]
            if len(v) == 0: continue
            v = (v - live[:, 0].min()).double() / 1000
            mins.append(v.min().item()); meds.append(v.median().item()); maxs.append(v.max().item())
        if mins: rows[k] = [round(statistics.median(x), 2) for x in (mins, meds, maxs)]
    # per-CTA atomic latency (10 - 9) and fence latency (12 - 9)
    lat = [((l[:, 10] - l[:, 9]).double() / 1000).median().item() for l in st if (l[:, 10] > 0).any()]
    print('ROW', json.dumps(dict(case=cn, probe=os.environ.get('B1_PROBE'), ok=ok, ev=round(statistics.median(ev), 3),
          ncta=len(st[-1]), atom_lat=round(statistics.median(lat), 3) if lat else None, stamps=rows)), flush=True)
print('DONE', flush=True)
