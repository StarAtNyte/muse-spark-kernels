"""Load geomean from judge per-shape times (stack_r2 check d835372941034b88) x Modal paired ratios (median over seeds).
usage: summ.py LOG [variant ...]"""
import json, sys, statistics, math
J = {'d1s': 4.69, 'd1f': 10.29, 'd16s': 7.46, 'd16f': 86.97, 'd32s': 9.31, 'd32f': 166.10,
     'v1s': 6.01, 'v1f': 12.55, 'v16s': 9.89, 'v16f': 97.35, 'v32s': 12.54, 'v32f': 178.82}
rows = [json.loads(l[4:]) for l in open(sys.argv[1]) if l.startswith('ROW ')]
vs = sys.argv[2:] or sorted({k for r in rows for k in r['gain'] if k != 'base'})
for v in vs:
    g = {}
    for r in rows:
        if v in (r.get('gain') or {}): g.setdefault(r['case'], []).append(r['gain'][v])
    gm = {c: statistics.median(x) for c, x in g.items()}
    loads = []
    for q in 'dv':
        for b in (1, 16, 32):
            s, f = f'{q}{b}s', f'{q}{b}f'
            t0 = 46 * J[s] + 16 * J[f]
            t1 = 46 * J[s] / gm.get(s, 1.0) + 16 * J[f] / gm.get(f, 1.0)
            loads.append(t0 / t1)
    print(v, 'load geomean %.4f' % math.exp(sum(map(math.log, loads)) / 6), {c: round(x, 4) for c, x in sorted(gm.items())},
          'loads', [round(x, 4) for x in loads])
