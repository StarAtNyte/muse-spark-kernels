"""Estimate local attention-score sensitivity using official paired-ratio records."""
import collections
import json
from pathlib import Path

p = Path('experiments/attention_readable_fp32_path_v1/status.json')
data = json.loads(p.read_text())
rows = [r for r in data['result']['records'] if r['event'] == 'benchmark']
groups = collections.defaultdict(list)
for r in rows:
    c = r['case']
    groups[c['load']].append(r)
assert len(groups) == 6 and all(len(rs) == 2 for rs in groups.values())
print('Candidate: attention_readable_fp32_path_v1; official private check 12167249d81c46ee')
print('Assumes other cases fixed; independent per-shape improvement, not a measured speedup.')
print('| Load | Layer | Time µs | Lower bound µs | Score gain for 1 µs saved | Score gain for 10% case speedup |')
print('|---|---|---:|---:|---:|---:|')
for name, rs in groups.items():
    total = sum(r['case']['weight'] * r['baseline_ns'] / r['speedup'] for r in rs)
    for r in rs:
        w = r['case']['weight']; time = r['baseline_ns'] / r['speedup']; floor = r['speed_of_light_ns']
        def gain(saved):
            return 100 * ((total / (total - w * saved)) ** (1 / len(groups)) - 1)
        print(f"| {name} | {r['case']['layer']} | {time / 1000:.2f} | {floor / 1000:.2f} | {gain(1000):.3f}% | {gain(time / 11):.3f}% |")
