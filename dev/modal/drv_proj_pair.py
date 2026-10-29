"""Projection paired screen: scored source (/w/kernels) vs candidate (/w/run/sub.py) on the given (m, layer) shapes
(env SHAPES, json list of [m, layer]); reference-checked; alternating cold-L2 graph replays; 3 seeds."""
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
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
B = load('base', '/w/kernels/muse_bf16_proj/submission.py'); C = load('cand', '/w/run/sub.py')
W = R.generate_weights(R.WEIGHTS_SEED)
sb, sc = B.prepare(W), C.prepare(W)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, st, d):
    o = M.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
for m, layer in json.loads(os.environ.get('SHAPES', '[[32, "qkv_gate"], [16, "qkv_gate"], [32, "o"]]')):
    rb, rc = [], []
    for seed in (61, 62, 63):
        d = R.generate_input(m, layer, seed)
        gb, ob = cap(B, sb, d); gc, oc = cap(C, sc, d)
        if not torch.is_tensor(oc): print('ROW', json.dumps(dict(m=m, layer=layer, error='not a tensor (SGLang fallback stubbed)'))); break
        ok, msg = R.check_implementation(d, oc, W)
        if not ok: print('CHECK_FAIL', m, layer, seed, msg, flush=True)
        tb, tc = [], []
        for rep in range(30):
            if rep % 2: tc.append(t1(gc)); tb.append(t1(gb))
            else: tb.append(t1(gb)); tc.append(t1(gc))
        rb.append(statistics.median(tb)); rc.append(statistics.median(tc))
    if rb:
        gb_, gc_ = statistics.geometric_mean(rb), statistics.geometric_mean(rc)
        print('ROW', json.dumps(dict(m=m, layer=layer, base=round(gb_, 2), cand=round(gc_, 2), gain=round(gb_ / gc_, 4))), flush=True)
print('DONE', flush=True)
