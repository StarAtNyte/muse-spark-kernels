def _fd_lab():
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')

    def timeit(fn, reps=10):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(reps):
            flush.add_(1)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    for batch, layer, spread in ((1, 'full', 'fixed'), (1, 'sliding', 'fixed'), (16, 'full', 'agentic'), (32, 'full', 'agentic'),
                                 (16, 'sliding', 'agentic'), (32, 'sliding', 'agentic')):
        data = _lab_data(batch, 8, layer, spread, seed=batch * 10 + 8)
        qkv, q = _fd_prepare(data)
        ref = _v7_kernel_fn(data).float()
        t7 = timeit(lambda: _v7_kernel_fn(data))
        splits_l = (8, 16, 24, 32, 48, 64, 96, 128, 192) if layer == 'full' else (1, 2, 4, 8, 16)
        res = []
        for stages in (2, 3, 4):
            for splits in splits_l:
                try:
                    t = timeit(lambda: _fd_forward(data, q, qkv, stages, splits, variant=5))
                    res.append((t, stages, splits))
                except Exception as ex:
                    pass
        res.sort()
        bs, bp = (res[0][1], res[0][2]) if res else (2, 32)
        o4 = _fd_forward(data, q, qkv, bs, bp, variant=5).float()
        err = ((o4 - ref).norm() / ref.norm()).item()
        print('LABA7', 'B=%d L=8 %s' % (batch, layer), 'v7=%.1f' % t7, 'chk4=%.2e' % err, ' '.join('%.1f:s%d/p%d' % r for r in res[:6]), flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()
