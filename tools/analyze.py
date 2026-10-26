#!/usr/bin/env python3
"""Summarize official per-shape timings using OKC's weighted load objective."""
import argparse
import collections
import json
import math
from pathlib import Path


def summarize(path):
    payload = json.loads(path.read_text())
    result = payload.get('result') or {}
    records = result.get('records', [])
    benches = [r for r in records if r.get('event') == 'benchmark']
    if not benches:
        return f'{path.parent.name}: {payload.get("status")} (no timing records)\n'
    lines = [f'# {path.parent.name}', '',
             f'Job: `{payload.get("id", payload.get("job_id"))}`',
             f'Correctness: {result.get("passed")}; official speedup: '
             f'{result.get("speedup", 0):.5f}x; CI: {result.get("speedup_ci")}', '',
             '| Load | Layer | Baseline µs | Candidate µs | Speedup | Floor µs |',
             '|---|---|---:|---:|---:|---:|']
    loads = collections.defaultdict(lambda: [0., 0., 0.])
    for r in benches:
        c = r['case']
        b, t, floor = r['baseline_ns']/1000, r['median_ns']/1000, r['speed_of_light_ns']/1000
        weight = c.get('weight', 1)
        sums = loads[c['load']]
        for j, val in enumerate((b, t, floor)):
            sums[j] += val * weight
        layer = c.get('layer', f'experts ({c.get("sharding", "")})')
        lines.append(f'| {c["load"]} | {layer} | {b:.2f} | {t:.2f} | {b/t:.3f}x | {floor:.2f} |')
    lines += ['', '| Load | Baseline ms | Candidate ms | Speedup | Ideal roofline bound |',
              '|---|---:|---:|---:|---:|']
    ratios = []
    for name, (b, t, floor) in loads.items():
        ratios.append(b/t)
        lines.append(f'| {name} | {b/1000:.3f} | {t/1000:.3f} | {b/t:.3f}x | {b/floor:.3f}x |')
    aggregate = math.exp(sum(math.log(x) for x in ratios)/len(ratios))
    lines += ['', f'Recomputed geometric mean across loads: {aggregate:.6f}x.',
              'Roofline bounds omit launch, merge, redundant traffic, and other costs.', '',
              '## Official profiler reports', '']
    for r in records:
        if r.get('event') == 'profile':
            c = r['case']
            layer = c.get('layer', f'experts ({c.get("sharding", "")})')
            lines += [f'### {c["load"]}: {layer}', '', '```text', r['report'], '```', '']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    path = args.experiment / 'status.json'
    report = summarize(path)
    out = args.experiment / 'analysis.md'
    out.write_text(report)
    # The full profiler appendix is retained in the artifact, not flooded into logs.
    print(report.split('## Official profiler reports')[0])
    print(f'Full report: {out}')


if __name__ == '__main__':
    main()
