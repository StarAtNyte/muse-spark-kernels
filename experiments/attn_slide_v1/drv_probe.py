"""Per-CTA globaltimer stamps of /w/modal/_slide/<v>.py probe builds, cold-L2 graph replays (diagnostic)."""
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
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
SPEC = {}
for q, qn in ((1, 'd'), (8, 'v')):
    for bsz in (1, 16, 32):
        SPEC[f'{qn}{bsz}s'] = dict(batch=bsz, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=q,
                                   spread=('agentic' if bsz > 1 else 'fixed'))
def graph(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for vn in os.environ.get('VARIANTS', 'probe').split(','):
    sp = importlib.util.spec_from_file_location('m_' + vn, f'/w/modal/_slide/{vn}.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
    for cn in os.environ.get('CASES', 'd1s,d16s,d32s,v1s,v16s,v32s').split(','):
        d = R.generate_input(**SPEC[cn], seed=11)
        dbg = torch.zeros(8192 * 16, dtype=torch.int64, device='cuda')
        ovr = json.loads(os.environ.get('OVR', '{}')).get(cn, {})
        o = S._launch(d, dbg=dbg, **ovr); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o))
        g0 = graph(lambda: S.kernel_fn(d)); gs = graph(lambda: S._launch(d, dbg=dbg, **ovr))
        ev, st = [], []
        for rep in range(15):
            ev.append(t1(g0)); dbg.zero_(); t1(gs)
            a = dbg.view(-1, 16).cpu(); st.append(a[a[:, 0] > 0])
        rows = {}
        for k in (0, 1, 2, 3, 4, 10, 11, 12, 9, 5, 6, 8, 7):
            mins, meds, maxs = [], [], []
            for live in st:
                v = live[:, k]; v = v[v > 0]
                if len(v) == 0: continue
                v = (v - live[:, 0].min()).double() / 1000
                mins.append(v.min().item()); meds.append(v.median().item()); maxs.append(v.max().item())
            if mins: rows[k] = [round(statistics.median(x), 2) for x in (mins, meds, maxs)]
        # per-CTA phase durations (median CTA, median rep)
        dur = {}
        for a_, b_ in ((0, 1), (1, 3), (3, 4), (10, 11), (11, 12), (4, 5), (5, 9), (9, 6), (5, 6), (6, 7)):
            vals = []
            for l in st:
                m = (l[:, a_] > 0) & (l[:, b_] > 0)
                if m.any(): vals.append(((l[m][:, b_] - l[m][:, a_]).double() / 1000).median().item())
            if vals: dur[f'{a_}-{b_}'] = round(statistics.median(vals), 2)
        print('ROW', json.dumps(dict(v=vn, ovr=ovr, case=cn, ok=ok, ev=round(statistics.median(ev), 3), ncta=len(st[-1]),
              stamps=rows, dur=dur)), flush=True)
print('DONE', flush=True)
