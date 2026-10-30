"""Lossless grouped scale format feasibility, independent of GPU runtime."""
import random

BASE = 0x50
MAX = BASE + 63

def pack(values):
    if len(values) != 8 or any(v < BASE or v > MAX for v in values):
        return None  # caller must retain the original bytes
    value = sum((v - BASE) << (6 * i) for i, v in enumerate(values))
    return value.to_bytes(6, 'little')

def unpack(data):
    v = int.from_bytes(data, 'little')
    return [BASE + ((v >> (6 * i)) & 63) for i in range(8)]

def scale_pair(packed, pair):
    # Original fast decode: BF16 bits = (encoded scale << 4) + 0x5c00.
    v = int.from_bytes(packed, 'little')
    a = ((v >> (12 * pair)) & 63) + BASE
    b = ((v >> (12 * pair + 6)) & 63) + BASE
    return (a * 16 + 0x5c00) | ((b * 16 + 0x5c00) << 16)

for v in range(256):
    p = pack([v] * 8)
    assert (p is not None) == (BASE <= v <= MAX)
    if p is not None:
        assert unpack(p) == [v] * 8
        assert scale_pair(p, 0) == ((v * 16 + 0x5c00) * 0x10001)
rng = random.Random(410)
for _ in range(100000):
    a = [rng.randrange(BASE, MAX + 1) for _ in range(8)]
    p = pack(a)
    assert unpack(p) == a
    for i in range(4):
        expected = (a[2*i]*16 + 0x5c00) | ((a[2*i+1]*16 + 0x5c00) << 16)
        assert scale_pair(p, i) == expected
print('PASS: exhaustive uniform bytes, 100000 random groups, all scale pairs')
print('8 scales: 8 -> 6 bytes; 32KB+4KB stage: 36KB -> 35KB (2.78% less)')
