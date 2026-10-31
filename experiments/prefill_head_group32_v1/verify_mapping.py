"""Check grouped query/head mapping for all grid indices and query-block counts."""
for nqb in range(1, 1025):
    full = nqb // 32 * 32
    rem = nqb - full
    seen = set()
    for linear in range(2 * nqb):
        tail = linear - 2 * full
        qi = (linear // 64) * 32 + (linear & 31) if linear < 2 * full else full + tail % max(rem, 1)
        hi = (linear >> 5) & 1 if linear < 2 * full else tail // max(rem, 1)
        seen.add((nqb - 1 - qi, hi))
    assert seen == {(q, h) for q in range(nqb) for h in (0, 1)}, nqb
print('PASS: grouped mapping is a bijection for nqb=1..1024')
