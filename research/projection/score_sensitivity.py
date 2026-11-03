"""Sensitivity of the official projection score to one independently improved case."""
import collections
import json
from pathlib import Path

p = Path('experiments/projection_fp32_splits_v1/status.json')
data = json.loads(p.read_text())
rows = [r for r in data['result']['records'] if r['event'] == 'benchmark']
groups = collections.defaultdict(list)
for r in rows:
    groups[r['case']['load']].append(r)
assert len(groups) == 6 and all(len(rs) == 5 for rs in groups.values())
print('Candidate: projection_fp32_splits_v1; private check 65ff350aa9884888')
print('Independent one-case estimates; no speedup is claimed. Lower bounds are theoretical.')
print('| Load | Layer | Time µs | Bound µs | Score gain for 1 µs saved | Score gain for 10% case speedup |')
print('|---|---|---:|---:|---:|---:|')
for name, rs in groups.items():
    total = sum(r['case']['weight'] * r['baseline_ns'] / r['speedup'] for r in rs)
    for r in rs:
        w = r['case']['weight']; t = r['baseline_ns'] / r['speedup']; floor = r['speed_of_light_ns']
        def gain(saved): return 100 * ((total / (total - w * saved)) ** (1 / len(groups)) - 1)
        print(f"| {name} | {r['case']['layer']} | {t / 1000:.2f} | {floor / 1000:.2f} | {gain(1000):.3f}% | {gain(t / 11):.3f}% |")
