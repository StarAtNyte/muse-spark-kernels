"""Summarize ROW lines: per-shape gain of each variant vs REF (median times), and the TASK.md load geomean
(load = 46 * sliding + 16 * full, per (q_len, batch); geomean over the 6 loads)."""
import sys, json, collections, math
ref = sys.argv[2]
med = collections.defaultdict(list)
for l in open(sys.argv[1]):
    if l.startswith('ROW'):
        d = json.loads(l[4:])
        for k, v in d['med'].items(): med[(d['case'], k)].append(v)
    elif l.startswith('TEST') or l.startswith('BUILDFAIL') or 'RETURNCODE' in l:
        print(l.strip()[:420])
names = sorted({k for _, k in med})
cases = sorted({c for c, _ in med}, key=lambda c: (c[0], int(c[1:-1]), c[-1]))
for c in cases:
    r = med[(c, ref)]
    print(c, ' '.join(f"{n}={statistics_mean:.4f}" for n, statistics_mean in
                     ((n, sum(a / b for a, b in zip(r, med[(c, n)])) / len(r)) for n in names if (c, n) in med)),
          ' ref_us=' + '/'.join(f'{x:.1f}' for x in r))
for n in names:
    lg = []
    for q in 'dv':
        for b in (1, 16, 32):
            cs, cf = f'{q}{b}s', f'{q}{b}f'
            if (cs, n) in med and (cf, n) in med and (cs, ref) in med:
                tr = 46 * sum(med[(cs, ref)]) / 3 + 16 * sum(med[(cf, ref)]) / 3
                tn = 46 * sum(med[(cs, n)]) / 3 + 16 * sum(med[(cf, n)]) / 3
                lg.append(tr / tn)
    if lg: print('LOADS', n, ' '.join(f'{x:.4f}' for x in lg), 'geomean', round(math.exp(sum(map(math.log, lg)) / len(lg)), 4))
