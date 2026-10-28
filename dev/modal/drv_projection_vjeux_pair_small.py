"""Paired projection comparison for shapes suggested by separate judge checks."""
import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch

class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec
        return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

base = load('/w/kernels/muse_bf16_proj/submission.py', 'base')
cand = load('/w/run/sub.py', 'candidate')
weights = R.generate_weights(98127)
bs = base.prepare(weights); cs = cand.prepare(weights)
flush = torch.empty(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)

def capture(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    g.replay(); torch.cuda.synchronize()
    return g, out

def elapsed(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000

for batch, layer in ((1, 'qkv_gate'), (1, 'router'), (1, 'unpooler'), (8, 'unpooler'), (256, 'router')):
    for seed in (419, 7919):
        data = R.generate_input(batch, layer, seed)
        gb, ob = capture(lambda: base.kernel_fn(data, bs))
        gc, oc = capture(lambda: cand.kernel_fn(data, cs))
        for label, out in (('base', ob), ('candidate', oc)):
            ok, msg = R.check_implementation(data, out, weights)
            if not ok: raise AssertionError((batch, layer, seed, label, msg))
        bt, ct, ratios = [], [], []
        for i in range(20):
            if i & 1: c, b = elapsed(gc), elapsed(gb)
            else: b, c = elapsed(gb), elapsed(gc)
            bt.append(b); ct.append(c); ratios.append(b / c)
        print('RESULT', json.dumps({'batch':batch,'layer':layer,'seed':seed,
              'base_us':statistics.median(bt),'candidate_us':statistics.median(ct),
              'base_over_candidate':statistics.median(ratios)}), flush=True)
        del data, gb, gc, ob, oc
