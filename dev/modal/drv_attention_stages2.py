"""Paired H200 screen of producer warp count on the active Gluon verify path."""
import importlib.util
import json
import statistics
import torch
from unittest.mock import MagicMock


class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = MagicMock()
        module.__path__ = []
        module.__name__ = spec.name
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


import sys
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('/w/kernels/muse_attention_decode/submission.py', 'base')
candidate = load('/w/run/sub.py', 'candidate')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def capture(module, batch, seed):
    data = R.generate_input(batch, 80000, 'tp8', 'full', seed,
                            spread='agentic', dist='coherent', page=1, q_len=8)
    module.kernel_fn(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.kernel_fn(data)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output, data


def elapsed(graph):
    flush.add_(1)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


for batch in (32,):
    for seed in (313, 9587, 3613):
        gb, ob, db = capture(base, batch, seed)
        for pr in (24,):
            candidate._os_FUSED_CFG[('full', 8)]['gv_pr'] = pr
            gc, oc, dc = capture(candidate, batch, seed)
            error = ((ob.float() - oc.float()).norm() / ob.float().norm()).item()
            if not torch.isfinite(oc).all().item() or error > 0.003:
                raise AssertionError((batch, seed, pr, error))
            ok, message = R.check_implementation(dc, R.collect(dc, oc))
            print("REFERENCE_CHECK", batch, seed, pr, ok, message, flush=True)
            if not ok:
                raise AssertionError((batch, seed, pr, message))
            bt, ct, ratios = [], [], []
            for rep in range(16):
                if rep % 2:
                    c, b = elapsed(gc), elapsed(gb)
                else:
                    b, c = elapsed(gb), elapsed(gc)
                bt.append(b)
                ct.append(c)
                ratios.append(b / c)
            print('RESULT', json.dumps(dict(batch=batch, seed=seed, stages=2,
                  relative_error=error, paired_speedup=statistics.median(ratios),
                  baseline_us=statistics.median(bt), candidate_us=statistics.median(ct))),
                  flush=True)
            del gc, oc, dc
        del gb, ob, db
print('GVPR_SCREEN_PASSED', torch.cuda.get_device_name(), flush=True)
