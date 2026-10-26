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

    for batch, layer, spread in ((16, 'full', 'agentic'), (32, 'full', 'agentic'), (32, 'sliding', 'agentic')):
        data = _lab_data(batch, 1, layer, spread, seed=batch * 10 + 1)
        qkv, q = _fd_prepare(data)
        cfgs = [(2, 64), (3, 32), (3, 64), (4, 32), (2, 128), (4, 64)] if layer == 'full' else [(3, 4), (3, 2), (2, 4)]
        for stages, splits in cfgs:
            row = []
            for dbg in (0, 1, 2, 4, 6, 7, 3, 5):
                _fd_module().fd2_set_dbg_t(dbg)
                try:
                    row.append('d%d=%.1f' % (dbg, timeit(lambda: _fd_forward(data, q, qkv, stages, splits, variant=2))))
                except Exception as ex:
                    row.append('d%d=ERR' % dbg)
            _fd_module().fd2_set_dbg_t(0)
            print('LABA2', 'B=%d %s s%d/p%d' % (batch, layer, stages, splits), ' '.join(row), flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()
