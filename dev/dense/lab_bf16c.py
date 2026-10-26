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
    shapes = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
    for layer, (n_full, k) in shapes.items():
        for m in (1, 8, 16, 32, 128, 256):
            n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
            x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
            base = state[layer][:n]
            cub = _dlab_time(lambda: x @ base.T, flush)
            kcs = k // 64
            for name, runner in (('U', _dense_run), ('C', _dense_crun)):
                res = []
                for ntb in (1, 2):
                    if (n // 16) % ntb:
                        continue
                    for nw in (2, 4, 8):
                        for ksplit, fix in ((1, 0), (2, 0), (4, 0), (8, 0), (2, 1), (4, 1), (8, 1), (16, 1)):
                            if kcs % (ksplit * nw):
                                continue
                            for nch in ((1,) if m <= 8 else (1, 2, 4) if m <= 32 else (2, 4)):
                                cfg = (ntb, nch, nw, 4, ksplit, fix)
                                try:
                                    t = _dlab_time(lambda: runner(x, state, layer, n, cfg), flush, 8)
                                except Exception:
                                    continue
                                res.append((t, cfg))
                res.sort()
                chk = 'na'
                if res:
                    o = runner(x, state, layer, n, res[0][1]).float()
                    ref = x.float() @ state[layer][:n].float().T
                    chk = '%.1e' % ((o - ref).norm() / ref.norm()).item()
                print('LABC', layer, 'm=%d' % m, name, 'chk=' + chk, 'cublas=%.1f' % cub, ' '.join('%.1f:%s' % (t, '-'.join(map(str, c))) for t, c in res[:4]), flush=True)
