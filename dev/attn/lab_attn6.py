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

    for batch, layer, spread, st, sp in ((16, 'full', 'agentic', 4, 64), (32, 'full', 'agentic', 4, 32), (32, 'sliding', 'agentic', 4, 4)):
        data = _lab_data(batch, 8, layer, spread, seed=batch * 10 + 8)
        qkv, q = _fd_prepare(data)
        row = []
        for dbg in (0, 1, 2, 4, 8, 3, 5, 6, 7, 9, 10, 12, 15):
            _fd_module().fd2_set_dbg_t(dbg)
            row.append('d%d=%.1f' % (dbg, timeit(lambda: _fd_forward(data, q, qkv, st, sp, variant=4))))
        _fd_module().fd2_set_dbg_t(0)
        print('LABA6', 'B=%d L=8 %s s%d/p%d' % (batch, layer, st, sp), ' '.join(row), flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()
