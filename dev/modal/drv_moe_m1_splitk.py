"""Paired cold-L2 H200 screen for the single-token split-K MoE path."""
import importlib.util
import json
import statistics
import sys
from unittest.mock import MagicMock

import torch


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


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

# Match the rank-local 32-expert tensor footprint used by the other H200 MoE
# screens, while generating route IDs from all 256 global experts.
R.N_EXPERTS = 32
weights = R.generate_weights(93819)
R.N_EXPERTS = 256
state = sub.prepare(weights)
base_fn = sub._sk1_prev_kernel_fn
candidate_fn = sub.kernel_fn
print('READY', torch.cuda.get_device_name(), 'grid', state['grid'], flush=True)

# The official replay flush is about 1 GiB; keep it outside the event interval.
flush = torch.empty(1024 * 1024 * 1024 // 4, dtype=torch.int32, device='cuda')


def capture(fn, data):
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data, state)
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def elapsed(graph):
    flush.add_(1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


results = []
for seed in (419, 2387, 6113, 11003, 18521, 31013):
    data = R.generate_input(1, 'ep8', seed)
    base_graph, base_out = capture(base_fn, data)
    candidate_graph, candidate_out = capture(candidate_fn, data)
    ok, message = R.check_implementation(data, candidate_out, weights)
    error = ((candidate_out.float() - base_out.float()).norm() /
             base_out.float().norm().clamp_min(1e-20)).item()
    print('CHECK', seed, ok, message, 'parent_rel_l2', error,
          'routes', data['topk_ids'].tolist(), flush=True)
    if not ok:
        raise AssertionError(message)

    ratios, base_times, candidate_times = [], [], []
    for repeat in range(16):
        if repeat & 1:
            candidate_us, base_us = elapsed(candidate_graph), elapsed(base_graph)
        else:
            base_us, candidate_us = elapsed(base_graph), elapsed(candidate_graph)
        ratios.append(base_us / candidate_us)
        base_times.append(base_us)
        candidate_times.append(candidate_us)
    result = dict(seed=seed, speedup=statistics.median(ratios),
                  baseline_us=statistics.median(base_times),
                  candidate_us=statistics.median(candidate_times),
                  relative_error=error)
    results.append(result)
    print('RESULT', json.dumps(result), flush=True)

aggregate = statistics.geometric_mean([r['speedup'] for r in results])
print('AGGREGATE', json.dumps(dict(speedup=aggregate, seeds=len(results))), flush=True)
