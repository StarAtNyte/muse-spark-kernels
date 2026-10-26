@triton.jit
def _skinny_matmul(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   SPLIT: tl.constexpr):
    rows = tl.arange(0, BM)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    kk = split * (K // SPLIT) + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0., tl.float32)
    for block in range(K // SPLIT // BK):
        k = kk + block * BK
        x = tl.load(X + rows[:, None] * K + k[None, :], rows[:, None] < M, 0.)
        w = tl.load(W + cols[:, None] * K + k[None, :], cols[:, None] < N, 0.)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(P + split * M * N + rows[:, None] * N + cols[None, :],
             acc, (rows[:, None] < M) & (cols[None, :] < N))
