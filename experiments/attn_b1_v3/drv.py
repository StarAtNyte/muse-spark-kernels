"""Paired A/B: parent (/w/modal/_b1lat/base.py) vs candidate (/w/run/sub.py) kernel_fn, plus b1 variants
(env VAR = {name: knobs for _b1_launch}). Cold-L2 graph replays, order rotated each rep, every graph
reference-checked. PROF=1: per-CTA globaltimer stamps of the candidate's b1 kernel (profiling build)."""
import importlib.util, json, os, statistics, sys, time
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
t0 = time.time()
NOBASE = os.environ.get('NOBASE') == '1'
if not NOBASE:
    C._MOD = B._module()   # identical parent CUDA source: share the build
print('parent build', round(time.time() - t0, 1), flush=True); t0 = time.time()
C._b1_module(); print('b1 build', round(time.time() - t0, 1), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    o = fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
SPEC = {
  'd1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1),
  'd1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=1),
  'v1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=8),
  'v1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=8),
  't1': dict(batch=1, ctx=10, heads='tp8', layer='sliding'),
  't1v': dict(batch=1, ctx=10, heads='tp8', layer='sliding', q_len=8),
  't9': dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
  't1e': dict(batch=1, ctx=6000, heads='tp8', layer='sliding', dist='edges', page=1),
  't1f': dict(batch=1, ctx=3000, heads='tp8', layer='full', kv_scale=0.015625),
  't1fe': dict(batch=1, ctx=6000, heads='tp8', layer='full', q_len=8, dist='edges'),
  'tv8': dict(batch=1, ctx=3000, heads='tp8', layer='sliding', q_len=8, dist='edges'),
  'tv8k': dict(batch=1, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=0.015625),
  't2047': dict(batch=1, ctx=2047, heads='tp8', layer='sliding'),
  'tf100k': dict(batch=1, ctx=100000, heads='tp8', layer='full', q_len=8, page=1),
  'd16s': dict(batch=16, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1, spread='agentic'),
}
VAR = json.loads(os.environ.get('VAR', '{}'))
REPS = int(os.environ.get('REPS', '24'))
PROF = os.environ.get('PROF') == '1'
for cn in os.environ.get('CASES', 'd1s,v1s,d1f,v1f').split(','):
    for seed in (11, 4242, 777)[:int(os.environ.get('SEEDS', '3'))]:
        d = R.generate_input(**SPEC[cn], seed=seed)
        gs, bad, err = {}, {}, {}
        fns = [] if NOBASE else [('base', lambda: B.kernel_fn(d))]
        fns += [('cand', lambda: C.kernel_fn(d))]
        fns += [(k, (lambda ov=v: C._b1_launch(d, **ov))) for k, v in VAR.items()]
        for vn, fn in fns:
            try:
                g, o = cap(fn); ok, msg = R.check_implementation(d, R.collect(d, o))
                err[vn] = str(msg)[:60]
                if ok: gs[vn] = g
                else: bad[vn] = str(msg)[:160]
            except Exception as e:
                bad[vn] = 'ERR ' + str(e)[:300]
        names = list(gs); ts = {k: [] for k in names}
        for rep in range(REPS):
            r = rep % len(names) if names else 0; order = names[r:] + names[:r]
            if rep % 2: order = order[::-1]
            for k in order: ts[k].append(t1(gs[k]))
        med = {k: round(statistics.median(v), 3) for k, v in ts.items()}
        ref = 'base' if 'base' in med else 'cand'
        gain = {k: round(med[ref] / v, 4) for k, v in med.items() if ref in med}
        print('ROW', json.dumps(dict(case=cn, seed=seed, med=med, gain=gain, bad=bad, err=err)), flush=True)
        if PROF and seed == 11:
            dbg = torch.zeros(256 * 16, dtype=torch.int64, device='cuda')
            kn = json.loads(os.environ.get('PROFKN', '{}'))
            gp, _ = cap(lambda: C._b1_launch(d, dbg=dbg, flags=('-DB1_PROF=1',), **kn))
            st = []
            for rep in range(12):
                dbg.zero_(); t1(gp); a = dbg.view(-1, 16).cpu(); st.append(a[a[:, 0] > 0])
            rows = {}
            for k in range(6):
                mins, meds, maxs = [], [], []
                for live in st:
                    v = live[:, k]; v = v[v > 0]
                    if len(v) == 0: continue
                    v = (v - live[:, 0].min()).double() / 1000
                    mins.append(v.min().item()); meds.append(v.median().item()); maxs.append(v.max().item())
                if mins: rows[k] = [round(statistics.median(x), 2) for x in (mins, meds, maxs)]
            print('PROF', json.dumps(dict(case=cn, ncta=len(st[-1]), stamps=rows)), flush=True)
print('DONE', flush=True)
if os.environ.get('SASS') == '1':
    import subprocess, re, collections
    out = subprocess.run(['cuobjdump', '-sass', C._b1_module().__file__], capture_output=True, text=True).stdout
    for fn in re.split(r'\n\s*Function : ', out)[1:]:
        ops = collections.Counter(re.findall(r'/\*[0-9a-f]{4,6}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]+)', fn))
        print('SASS', fn.split('\n', 1)[0][-40:], sum(ops.values()), dict(ops.most_common(14)), flush=True)
