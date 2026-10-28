"""Compare complete official attention checks; per-case leads require hybrid validation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate');args=p.parse_args()
def records(name):
 d=json.loads((Path('experiments')/name/'status.json').read_text())
 assert d['status']=='done',(name,d['status'])
 r=d.get('result') or {};assert r.get('passed'),(name,r.get('error'))
 rr=[x for x in r['records'] if x.get('event')=='benchmark']
 assert len(rr)==12,(name,len(rr));return rr
old=records('attention_fp8p_cluster_v1');new=records(args.candidate)
def key(r):
 c=r['case'];return c['layer'],c['batch'],c.get('q_len',8 if c['load'].startswith('verify') else 1),c['ctx'],c['page']
lookup={key(r):r for r in old};assert set(lookup)=={key(r) for r in new}
print('| Layer | Batch | Q | Old us | New us | Ratio of baseline-normalized speedups |')
print('|---|---:|---:|---:|---:|---:|')
for r in new:
 k=key(r);o=lookup[k];ratio=r['paired_ratio']/o['paired_ratio']
 print(f'| {k[0]} | {k[1]} | {k[2]} | {o["median_ns"]/1000:.2f} | {r["median_ns"]/1000:.2f} | {ratio:.3f}x |')
print('Different runs use fresh inputs; ratios identify candidate paths, not isolated causal gains. Validate combined dispatch end to end.')
