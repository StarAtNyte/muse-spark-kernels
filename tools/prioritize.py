#!/usr/bin/env python3
"""Rank shape changes by their effect on the official geometric mean of loads.

This predicts within-problem score impact, never overall leaderboard rank.
Reference-model estimates omit overhead and can be beaten; they are not hard ceilings.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def rank(problem, entry, reduction, status_path=None):
    path=Path(status_path) if status_path else ROOT/'experiments'/entry['experiment']/'status.json'
    status=json.loads(path.read_text()); result=status.get('result') or {}
    records=[r for r in result.get('records',[]) if r.get('event')=='benchmark']
    if status.get('status')!='done' or result.get('passed') is not True or not records:
        raise ValueError(f'{problem}: missing complete validated timings')
    totals=defaultdict(float);baselines=defaultdict(float)
    for r in records:
        c=r['case'];w=c.get('weight',1);totals[c['load']]+=r['median_ns']*w;baselines[c['load']]+=r['baseline_ns']*w
    count=len(totals);score=math.exp(sum(math.log(baselines[k]/v) for k,v in totals.items())/count)
    if not math.isclose(score,result['speedup'],rel_tol=1e-5):
        raise ValueError(f'{problem}: score reconstruction disagrees with official result')
    rows=[]
    for r in records:
        c=r['case'];w=c.get('weight',1);total=totals[c['load']];time=r['median_ns'];floor=r['speed_of_light_ns']
        predicted=(total/(total-time*w*reduction))**(1/count)
        model_gain=(total/(total-max(0,time-floor)*w))**(1/count)
        rows.append(dict(problem=problem,load=c['load'],layer=c.get('layer','experts'),
            current_us=time/1000,load_fraction=time*w/total,
            track_gain_percent=(predicted-1)*100,reference_model_gain_percent=(model_gain-1)*100))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--problem');p.add_argument('--reduction',type=float,default=.2);p.add_argument('--json',action='store_true');p.add_argument('--top',type=int,default=16);p.add_argument('--status',help='use a specific official status.json; requires --problem');a=p.parse_args()
    if not 0<a.reduction<1:p.error('--reduction must be between 0 and 1')
    best=json.loads((ROOT/'research/best.json').read_text());rows=[]
    if a.status and not a.problem:p.error('--status requires --problem')
    if a.problem and a.problem not in best and not a.status:p.error('unknown problem')
    if a.status:
        status=json.loads(Path(a.status).read_text())
        if status.get('board') != a.problem:p.error('--problem does not match status board')
        rows.extend(rank(a.problem, {}, a.reduction, a.status))
    else:
        for problem,entry in best.items():
            if not a.problem or a.problem==problem:rows.extend(rank(problem,entry,a.reduction))
    rows.sort(key=lambda r:-r['track_gain_percent'])
    if a.json:print(json.dumps(rows,indent=2));return
    print(f'Effect of reducing ONE shape latency by {100*a.reduction:g}%; gains refer to its own problem score.')
    print('| Problem | Load | Layer | Current us | Load share | Track gain | Reference-model gain |')
    print('|---|---|---|---:|---:|---:|---:|')
    for r in rows[:a.top]:print(f"| {r['problem']} | {r['load']} | {r['layer']} | {r['current_us']:.2f} | {r['load_fraction']:.1%} | {r['track_gain_percent']:.2f}% | {r['reference_model_gain_percent']:.2f}% |")
if __name__=='__main__':main()
