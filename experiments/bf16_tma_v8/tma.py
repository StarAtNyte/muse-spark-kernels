@triton.jit
def _tma_matmul(A, B, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SPLIT: tl.constexpr):
    mi, ni, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = split * (K // SPLIT)
    acc = tl.full((BM, BN), 0., tl.float32)
    for block in range(K // SPLIT // BK):
        kk = start + block * BK
        a = A.load([mi * BM, kk])
        b = B.load([ni * BN, kk])
        acc = tl.dot(a, tl.trans(b), acc)
    rows = mi * BM + tl.arange(0, BM)
    cols = ni * BN + tl.arange(0, BN)
    tl.store(P + split * M * N + rows[:, None] * N + cols[None, :],
             acc, (rows[:, None] < M) & (cols[None, :] < N))
