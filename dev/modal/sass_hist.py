import re, collections, sys
fn = sys.argv[1]
L = open(fn).read().split('\n')
st = [i for i, l in enumerate(L) if 'Function : ' in l and sys.argv[2] in l]
for s in st:
    en = [i for i, l in enumerate(L) if 'Function :' in l and i > s]
    en = en[0] if en else len(L)
    ops = collections.Counter()
    for l in L[s:en]:
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)', l)
        if m: ops[m.group(2)] += 1
    print(L[s].strip()[:90], sum(ops.values())); print(ops.most_common(28))
