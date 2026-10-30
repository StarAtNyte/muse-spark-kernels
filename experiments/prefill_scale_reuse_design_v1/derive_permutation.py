"""Exhaustively verify an abstract K64 scale-reuse permutation.

This proves only index arithmetic, not the prefill kernel fragment layout or numeric correctness.
"""
from fractions import Fraction

def original_index(p):
    step, c = divmod(p, 16)
    lane_group = (c >> 1) & 3
    within = (c & 1) | ((c >> 3) << 1)
    return 16 * lane_group + 4 * step + within

perm = [original_index(p) for p in range(64)]
assert sorted(perm) == list(range(64))
inverse = [0] * 64
for p, k in enumerate(perm):
    inverse[k] = p
assert all(perm[inverse[k]] == k for k in range(64))
for c in range(16):
    positions = [perm[16 * step + c] for step in range(4)]
    assert len({k // 16 for k in positions}) == 1
    assert len(set(positions)) == 4
# Any x and row of weights, their consistently permuted dot product is exact algebraically.
x = [Fraction((k * 7) % 19 - 9, 7) for k in range(64)]
w = [Fraction((k * 11) % 23 - 11, 5) for k in range(64)]
scales = [Fraction(1, 3), Fraction(-2, 5), Fraction(0), Fraction(7, 4)]
original = sum(x[k] * w[k] * scales[k // 16] for k in range(64))
permuted = sum(x[perm[p]] * w[perm[p]] * scales[perm[p] // 16] for p in range(64))
assert original == permuted
print('PASS: 64-index bijection, inverse, four-step scale reuse, exact algebraic dot product')
print('position -> source index:', perm)
