def _dlab_time(fn, flush, reps=12):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
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


def _dense_lab(state):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    mod = state['dense_mod']
    out = torch.zeros(4, dtype=torch.float32, device='cuda')
    for mb in (38, 67):
        buf = torch.randint(0, 255, (mb * 1024 * 1024,), dtype=torch.uint8, device='cuda')
        res = []
        for blocks, threads in ((132 * 2, 256), (132 * 4, 256), (132 * 8, 256), (132 * 16, 256), (132 * 4, 512), (132 * 2, 1024)):
            res.append('%dx%d=%.1f' % (blocks, threads, _dlab_time(lambda: mod.probe_t(buf, out, blocks, threads), flush)))
        print('LABP read %dMB' % mb, ' '.join(res), flush=True)
        del buf
    # trivial kernel: fixed cost of a graph with one tiny launch
    tiny = torch.zeros(1, device='cuda')
    print('LABP tiny', '%.1f' % _dlab_time(lambda: tiny.add_(1), flush), flush=True)
    cases = (('qkv_gate', 2304, 8192, 16, (2, 2, 8, 4, 1, 0)), ('qkv_gate', 2304, 8192, 32, (2, 4, 4, 4, 1, 0)),
             ('pooler', 4096, 8192, 32, (2, 4, 4, 4, 1, 0)), ('o', 8192, 1024, 16, (2, 2, 8, 4, 1, 0)),
             ('unpooler', 8192, 4096, 32, (2, 4, 4, 4, 1, 0)), ('qkv_gate', 2304, 8192, 128, (2, 4, 4, 4, 1, 0)))
    for layer, n, k, m, cfg in cases:
        x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
        base = state[layer][:n]
        cub = _dlab_time(lambda: x @ base.T, flush)
        row = []
        for d in (0, 1, 2, 3):
            mod.dense_set_dbg_t(d)
            try:
                row.append('d%d=%.1f' % (d, _dlab_time(lambda: _dense_run(x, state, layer, n, cfg), flush)))
            except Exception as ex:
                row.append('d%d=ERR' % d)
        mod.dense_set_dbg_t(0)
        print('LABD2', layer, 'm=%d' % m, 'cfg', cfg, 'cublas=%.1f' % cub, ' '.join(row), flush=True)
