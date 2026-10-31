"""Verify launch-index mapping and the unmodified split path on the CPU."""
for nqb in range(1, 1025):
    for nsplit in (1, 2, 4):
        old, new = set(), set()
        for by in range(2 * nsplit):
            for bx in range(nqb):
                old.add((nqb - 1 - bx, by & 1, by >> 1))
                linear = bx + by * nqb
                interleave = nsplit == 1
                new.add((nqb - 1 - (linear // 2 if interleave else bx),
                         linear & 1 if interleave else by & 1, by >> 1))
        assert old == new and len(new) == nqb * 2 * nsplit, (nqb, nsplit)
print('PASS: equivalent unique (query block, head, split) assignments for nqb=1..1024 and 1/2/4 splits')
