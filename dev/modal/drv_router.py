"""Cold-cache, paired CUDA graph router sweep; diagnostic, not official scoring."""
import importlib.util
import json
import statistics
import sys

import torch

sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
weights = R.generate_weights(98127)
state = sub._prepare_v5(weights)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, result


def elapsed(graph):
    flush.add_(1)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


print('GPU', torch.cuda.get_device_name(), 'torch', torch.__version__, flush=True)
for batch in (1, 8, 16, 32, 128, 256):
    data = R.generate_input(batch, 'router', 713 + batch)
    baseline, _ = capture(lambda: sub._kernel_v5(data, state))
    rows = []
    for cfg in sub._ROUTER_CFGS:
        try:
            graph, output = capture(lambda: sub._router_run(data['x'], state, 256, cfg))
            ok, message = R.check_implementation(data, output, weights)
            if not ok:
                raise ValueError(message)
            ratios, base_times, times = [], [], []
            for repeat in range(12):
                if repeat % 2:
                    candidate_us, base_us = elapsed(graph), elapsed(baseline)
                else:
                    base_us, candidate_us = elapsed(baseline), elapsed(graph)
                ratios.append(base_us / candidate_us)
                base_times.append(base_us)
                times.append(candidate_us)
            row = dict(batch=batch, cfg=cfg, paired_speedup=statistics.median(ratios),
                       baseline_us=statistics.median(base_times), candidate_us=statistics.median(times))
            rows.append(row)
            print('RESULT', json.dumps(row), flush=True)
        except Exception as exc:
            print('FAIL', batch, cfg, str(exc)[:250], flush=True)
    rows.sort(key=lambda row: row['paired_speedup'], reverse=True)
    print('BEST', json.dumps(rows[:3]), flush=True)

# Exercise outliers and nearby batch sizes with every configuration, not just winners.
for batch, dist in ((5, 'normal'), (9, 'outliers'), (33, 'normal'), (130, 'outliers')):
    data = R.generate_input(batch, 'router', 991 + batch, dist=dist)
    for cfg in sub._ROUTER_CFGS:
        output = sub._router_run(data['x'], state, 256, cfg)
        ok, message = R.check_implementation(data, output, weights)
        if not ok:
            raise AssertionError((batch, dist, cfg, message))
    print('COVERAGE', batch, dist, 'all configurations passed', flush=True)
