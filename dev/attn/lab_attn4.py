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
        data = _lab_data(batch, 1, layer, spread, seed=batch * 10 + 1)
        qkv, q = _fd_prepare(data)
        ref = _fd_forward(data, q, qkv, 3, 32 if layer == 'full' else 4, variant=2).float()
        splits_l = (16, 32, 48, 64, 96, 128, 192) if layer == 'full' else (1, 2, 4, 8, 16)
        res = []
        for variant in (2, 3):
            for stages in (2, 3, 4):
                for splits in splits_l:
                    try:
                        t = timeit(lambda: _fd_forward(data, q, qkv, stages, splits, variant=variant))
                        res.append((t, variant, stages, splits))
                    except Exception as ex:
                        pass
        res.sort()
        o3 = _fd_forward(data, q, qkv, 3, 32 if layer == 'full' else 4, variant=3).float()
        err = ((o3 - ref).norm() / ref.norm()).item()
        b2 = [r for r in res if r[1] == 2][:2]
        b3 = [r for r in res if r[1] == 3][:4]
        print('LABA4', 'B=%d %s' % (batch, layer), 'chk3=%.2e' % err, 'fd2:', ' '.join('%.1f:s%d/p%d' % (r[0], r[2], r[3]) for r in b2),
              'fd3:', ' '.join('%.1f:s%d/p%d' % (r[0], r[2], r[3]) for r in b3), flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()
