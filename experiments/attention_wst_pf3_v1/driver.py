"""Attention paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on the 12 scored shapes;
each output reference-checked; alternating cold-L2 graph replays."""
import importlib.util, json, statistics, sys
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
import ast, base64, zlib
from pathlib import Path
source=Path('/w/run/sub.py').read_text()
tree=ast.parse(source)
enc=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and len(n.args[0].value)>1000]
assert len(enc)==1
inner=zlib.decompress(base64.b64decode(enc[0])).decode()
a='launch_ct14b<4, 3, 4, 1>';b='launch_ct14b<4, 2, 4, 1>'
assert inner.count(a)==1
Path('/w/run/base_pf2.py').write_text(source.replace(enc[0],base64.b64encode(zlib.compress(inner.replace(a,b).encode(),9)).decode(),1))
B = load('base', '/w/run/base_pf2.py')
C = load('cand', '/w/run/sub.py')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, d):
    o = M.kernel_fn(d); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for trial in (0, 1, 2):
    for q in (8,):
        for bsz in (16, 32):
            for layer, dist in (('full', 'coherent'),):
                c = dict(batch=bsz, ctx=80000, heads='tp8', layer=layer, page=1, dist=dist, q_len=q,
                         spread=('agentic' if bsz > 1 else 'fixed'))
                d = R.generate_input(**c, seed=5000 + bsz + q + trial * 101)
                res = {}
                for name, M in (('ours', B), ('cand', C)):
                    try:
                        g, o = cap(M, d); ok, msg = R.check_implementation(d, R.collect(d, o)); res[name] = (g, ok, msg)
                    except Exception as e:
                        res[name] = (None, False, 'run failed: ' + str(e)[:120])
                        try: torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
                        except Exception: pass
                tag = f"{'verify' if q == 8 else 'decode'} {bsz} {layer}"
                if not all(v[0] is not None for v in res.values()):
                    print('ROW', json.dumps(dict(case=tag, error={k: v[2] for k, v in res.items()})), flush=True); continue
                ts = {k: [] for k in res}
                for rep in range(24):
                    for k in (('ours', 'cand') if rep % 2 == 0 else ('cand', 'ours')): ts[k].append(t1(res[k][0]))
                med = {k: statistics.median(v) for k, v in ts.items()}
                print('ROW', json.dumps(dict(case=tag, ours=round(med['ours'], 1), cand=round(med['cand'], 1),
                      cand_gain=round(med['ours'] / med['cand'], 3), ok={k: v[1] for k, v in res.items()})), flush=True)
print('DONE', flush=True)
