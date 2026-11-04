"""Paired per-shape gains (mean over ROW lines of base/variant) -> TASK.md load speedups using the judge's
per-shape times of the parent (job 979f4c2f587f4325); shapes not timed count as gain 1."""
import sys, json, collections, math
J = dict(d1s=5.20, d1f=11.18, d16s=7.96, d16f=86.62, d32s=10.14, d32f=166.48,
         v1s=6.42, v1f=13.33, v16s=10.44, v16f=97.17, v32s=13.14, v32f=179.19)
g = collections.defaultdict(list)
for fn in sys.argv[1:]:
    for l in open(fn):
        if l.startswith('ROW'):
            d = json.loads(l[4:])
            m = d['med']
            if 'base' in m:
                # judge-equivalent gain: the paired time delta applied to the judge's parent time
                for k, v in m.items(): g[(d['case'], k)].append(J[d['case']] / (J[d['case']] - (m['base'] - v)))
names = sorted({k for _, k in g if k != 'base'})
for n in names:
    per = {c: sum(g[(c, n)]) / len(g[(c, n)]) for c in J if (c, n) in g}
    lg = []
    for q in 'dv':
        for b in (1, 16, 32):
            s, f = f'{q}{b}s', f'{q}{b}f'
            gs, gf = per.get(s, 1.0), per.get(f, 1.0)
            lg.append((46 * J[s] + 16 * J[f]) / (46 * J[s] / gs + 16 * J[f] / gf))
    print(n, ' '.join(f'{c}={v:.3f}(n{len(g[(c, n)])})' for c, v in per.items()))
    print('  LOADS', ' '.join(f'{x:.4f}' for x in lg), 'geomean', round(math.exp(sum(map(math.log, lg)) / 6), 4))
