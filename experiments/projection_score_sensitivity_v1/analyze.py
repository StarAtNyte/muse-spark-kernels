"""Score sensitivity from published projection shape medians; no floor is assumed attainable."""
from pathlib import Path
import re, math
p=Path('experiments/scored_projection_vjeux_hybrid_v1/profiles/submission.beb3e26afb2544c9.txt')
rows=[]
for line in p.read_text().splitlines():
 m=re.match(r'layer=(\w+) load=(decode|verify) (\d+) m=(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s+([\d.]+)',line)
 if m:
  layer,kind,batch,ms,base,ours,sol=m.groups(); rows.append(dict(layer=layer,load=f'{kind} {batch}',m=int(ms),ours=float(ours),sol=float(sol)))
assert len(rows)==30
loads={}
for row in rows: loads[row['load']]=loads.get(row['load'],0)+row['ours']
for row in rows:
 total=loads[row['load']];delta=.1*row['ours'];row['pct_gain_10pct']=100*((total/(total-delta))**(1/6)-1)
 row['gap_us']=round(row['ours']-row['sol'],2)
print('Baseline: official scored job beb3e26afb2544c9; one shape improved 10%, other cases fixed.')
print('Case | our latency us | reference SOL us | gap us | projected score gain %')
for r in sorted(rows,key=lambda r:r['pct_gain_10pct'],reverse=True):
 print(f"{r['load']} {r['layer']} m={r['m']} | {r['ours']:.2f} | {r['sol']:.2f} | {r['gap_us']:.2f} | {r['pct_gain_10pct']:.3f}")
