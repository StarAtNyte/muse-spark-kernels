"""Non-flat attention pipeline depth (Cfg STAGES for split-KV grids) screen: variants built from the scored source
with only the STAGES constant changed; reference-checked; paired cold-L2 graph timing."""
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
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
old = '__sincosf(rr, &sv, &cv);'
assert S.CUDA_SRC.count(old) == 1
base_src = S.CUDA_SRC
S.CUDA_SRC = base_src.replace(old, 'sv = 0.f; cv = 1.f;')
mods = {'invalid_no_angles': S._build(tag='_no_rope_angles')}
S.CUDA_SRC = base_src
print('BUILT', list(mods), flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
CASES = [dict(batch=1, layer='sliding', dist='soft'),
         dict(batch=1, layer='sliding', dist='soft', q_len=8),
         dict(batch=16, layer='sliding', dist='soft', spread='agentic'),
         dict(batch=16, layer='sliding', dist='soft', spread='agentic', q_len=8),
         dict(batch=32, layer='sliding', dist='soft', spread='agentic'),
         dict(batch=32, layer='sliding', dist='soft', spread='agentic', q_len=8)]
for c in CASES:
    for seed in (11,):
        d = R.generate_input(ctx=80000, heads='tp8', page=1, seed=seed, **c)
        graphs = {}
        for name, mod in [('base', None)] + list(mods.items()):
            try:
                g, o = cap((lambda: S.kernel_fn(d)) if mod is None else (lambda m=mod: S._launch(d, _mod=m)))
                if name == 'base':
                    ok, msg = R.check_implementation(d, R.collect(d, o))
                    if not ok: print('CHECK_FAIL', name, c, msg, flush=True); continue
                else:
                    print('ABLATION_INVALID_OUTPUT', name, flush=True)
                graphs[name] = g
            except Exception as e:
                print('RUN_FAIL', name, c, str(e)[:120], flush=True)
                try: torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize()
                except Exception: pass
        names = list(graphs); ts = {k: [] for k in names}
        for rep in range(24):
            for k in (names if rep % 2 == 0 else names[::-1]): ts[k].append(t1(graphs[k]))
        med = {k: statistics.median(v) for k, v in ts.items()}
        print('ROW', json.dumps(dict(case=c, seed=seed, **{k: round(v, 2) for k, v in med.items()},
              **{f'gain_{k}': round(med['base'] / v, 3) for k, v in med.items() if k != 'base'})), flush=True)
print('DONE', flush=True)
