"""Correctness on the 9 tests + 30 scored cells, then cold-L2 graph-replay medians of the 30 scored cells."""
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
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
tag = os.environ['TAG']
spec = importlib.util.spec_from_file_location('sub', os.environ['SUB']); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
t0 = time.time(); st = S.prepare(w); print('PREP', tag, round(time.time() - t0, 1), flush=True)
tests = [(1, 'qkv_gate', 'normal'), (7, 'pooler', 'normal'), (33, 'o', 'normal'), (130, 'unpooler', 'normal'), (5, 'unpooler', 'normal'),
         (8, 'router', 'normal'), (9, 'router', 'outliers'), (512, 'qkv_gate', 'outliers'), (1000, 'unpooler', 'outliers')]
cells = [(m, l) for m in (1, 16, 32, 8, 128, 256) for l in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')]
bad = 0
for i, (m, l, dist) in enumerate(tests + [(m, l, 'normal') for m, l in cells]):
    for seed in (100 + i, 200 + i):
        d = R.generate_input(m, l, seed, dist=dist)
        out = S.kernel_fn(d, st); torch.cuda.synchronize()
        ok = R.check_implementation(d, out, w)
        if not ok[0]:
            bad += 1; print('FAIL', tag, m, l, dist, str(ok)[:200], flush=True)
print('CORRECT', tag, 'fails', bad, flush=True)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
REPS = int(os.environ.get('REPS', '60'))
for m, l in cells:
    d = R.generate_input(m, l, 777)
    S.kernel_fn(d, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    print('ROW', json.dumps({'tag': tag, 'm': m, 'layer': l, 'us': round(statistics.median(ts), 3)}), flush=True)
    del g
print('DONE', tag, flush=True)
