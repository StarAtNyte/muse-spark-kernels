"""Per-cell paired ratios (A/B, >1 = B faster) and the load-geomean change, judge-style (4 us fixed cost removed)."""
import json, math, statistics, sys
from collections import defaultdict
rows = defaultdict(lambda: defaultdict(list))
for line in open(sys.argv[1]):
    if line.startswith('ROW '):
        j = json.loads(line[4:]); rows[(j['m'], j['layer'])][j['tag'][0]].append(j['us'])
FIX = 4.0
loads = defaultdict(lambda: [0.0, 0.0])
for (m, l), d in sorted(rows.items()):
    a, b = statistics.median(d['A']), statistics.median(d['B'])
    loads[m][0] += a - FIX; loads[m][1] += b - FIX
    flag = '  <==' if abs(a / b - 1) > 0.01 else ''
    print(f"m{m:<4}{l:<9} A {a:7.2f} B {b:7.2f} A/B {a / b:.4f} judge-style {(a - FIX) / (b - FIX):.4f}  A {d['A']} B {d['B']}{flag}")
g = math.exp(sum(math.log(x[0] / x[1]) for x in loads.values()) / len(loads))
for m, (a, b) in sorted(loads.items()):
    print(f"load m{m}: A {a:.2f} B {b:.2f} ratio {a / b:.4f}")
print(f"BOARD geomean A/B = {g:.4f}")
