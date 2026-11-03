"""CPU sensitivity study, not a model of the hardware's approximate exp2."""
import bisect
import math
import struct

# Positive finite E4M3 values, excluding the NaN encoding. Round to nearest,
# ties to even, and saturate at the largest finite number.
vals = [(0.0 if c == 0 else (c & 7) * 2**-9 if c < 8 else (1 + (c & 7) / 8) * 2**((c >> 3) - 7), c) for c in range(0x7f)]
xs = [v for v, _ in vals]
def e4(x):
    if x >= xs[-1]: return vals[-1][1]
    i = bisect.bisect_left(xs, x)
    if i == 0: return 0
    a, b = xs[i-1], xs[i]
    return vals[i-1][1] if x-a < b-x else vals[i][1] if b-x < x-a else vals[i-1][1] if vals[i-1][1] % 2 == 0 else vals[i][1]
def half(x):
    try: return struct.unpack('<e', struct.pack('<e', x))[0]
    except OverflowError: return math.copysign(math.inf,x)

total = changed = 0
maxrel = 0.0
for i in range(200001):
    score = -24 + i * 32 / 200000
    exact = 2 ** score
    rounded_score = half(score)
    rounded_probability = half(2 ** rounded_score)
    expected, observed = e4(exact), e4(rounded_probability)
    total += 1
    changed += expected != observed
    if exact >= 2**-9:
        maxrel = max(maxrel, abs(rounded_probability / exact - 1))
print('score range [-24, 8], samples', total)
print('E4M3 codes changed', changed, 'fraction', changed / total)
print('max pre-E4M3 relative error when exact P >= 2^-9:', maxrel)
print('This models ideal exp2 with FP16 rounding, not ex2.approx.f16x2 or output error.')
