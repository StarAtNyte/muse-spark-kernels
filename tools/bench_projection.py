#!/usr/bin/env python3
"""Development-only, paired cold-L2 CUDA-graph timing against pinned SGLang.

Run inside the challenge's SGLang image on the target H200. This is diagnostic
timing, not an implementation of or substitute for the official score.
Imports judge reference files only in this development process, never in a
submission. All emitted times are raw graph timings without overhead removal.
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--layer', choices=['qkv_gate', 'o', 'pooler', 'unpooler', 'router'],
                        default='pooler')
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--pairs', type=int, default=30)
    parser.add_argument('--cold-mib', type=int, default=256)
    parser.add_argument('--output', type=Path, default=Path('local-benchmark.json'))
    args = parser.parse_args()
    if min(args.m, args.pairs, args.cold_mib) <= 0:
        parser.error('m, pairs, and cold-mib must be positive')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA and the challenge SGLang environment are required')
    root = Path(__file__).resolve().parents[1]
    challenge = root / 'kernels/muse_bf16_proj'
    sys.path.insert(0, str(challenge))
    candidate = load('dev_candidate', args.candidate.resolve())
    baseline = load('dev_baseline', challenge / 'baseline.py')
    reference = load('dev_reference', challenge / 'reference.py')
    weights = reference.generate_weights(args.seed)
    state = candidate.prepare(weights)
    baseline_state = baseline.prepare(weights)
    # Use fresh weights and multiple input distributions, including probe rows.
    for offset, dist in enumerate(('normal', 'outliers')):
        data = reference.generate_input(args.m, args.layer, args.seed + offset + 1, dist=dist)
        output = candidate.kernel_fn(data, state)
        ok, message = reference.check_implementation(data, output, weights)
        if not ok:
            raise RuntimeError(f'correctness failed ({dist}): {message}')
    data = reference.generate_input(args.m, args.layer, args.seed + 3)
    cold = torch.empty(args.cold_mib * 1024**2, device='cuda', dtype=torch.uint8)
    functions = {'candidate': lambda: candidate.kernel_fn(data, state),
                 'baseline': lambda: baseline.kernel_fn(data, baseline_state)}
    graphs, keepalive = {}, {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name, fn in functions.items():
            for _ in range(5):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                keepalive[name] = fn()
            graphs[name] = graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    times = {name: [] for name in graphs}
    ratios = []
    for pair in range(args.pairs):
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for name in order:
            # Flush is outside event timing but ordered before the replay.
            cold.fill_(pair % 251)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(begin.elapsed_time(end) * 1000)
        ratios.append(times['baseline'][-1] / times['candidate'][-1])
    result = dict(device=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(), torch=torch.__version__,
                  candidate=str(args.candidate.resolve()), layer=args.layer, m=args.m,
                  seed=args.seed, pairs=args.pairs, cold_mib=args.cold_mib,
                  timing='raw CUDA graph events; not official score', times_us=times,
                  median_us={name: statistics.median(t) for name, t in times.items()},
                  paired_speedup=statistics.median(ratios))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'times_us'}, indent=2))


if __name__ == '__main__':
    main()
