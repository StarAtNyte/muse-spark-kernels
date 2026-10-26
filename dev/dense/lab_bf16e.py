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


def _dense2_run(x, state, layer, n, cfg):
    nw, ntb, s, ks = cfg
    m = x.shape[0]
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    state['dense_mod'].dense2_gemm(nw, ntb, nch, s, ks, state['d_' + layer][: n // 16], x, out)
    return out


def _dense_lab(state):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    shapes = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
    for layer, (n_full, k) in shapes.items():
        for m in (8, 16, 32):
            n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
            x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
            base = state[layer][:n]
            ref = x.float() @ base.float().T
            cub = _dlab_time(lambda: x @ base.T, flush)
            res = []
            bad = 0
            for nw, ntb in ((8, 1), (4, 1), (4, 2)):
                for s in (4, 6):
                    for ks in (2, 4, 8):
                        cfg = (nw, ntb, s, ks)
                        try:
                            o = _dense2_run(x, state, layer, n, cfg).float()
                            torch.cuda.synchronize()
                        except Exception as ex:
                            continue
                        err = ((o - ref).norm() / ref.norm()).item()
                        if err > 5e-3:
                            bad += 1
                            continue
                        res.append((_dlab_time(lambda: _dense2_run(x, state, layer, n, cfg), flush, 8), cfg))
            res.sort()
            print('LABE', layer, 'm=%d' % m, 'cublas=%.1f' % cub, 'bad=%d' % bad, ' '.join('%.1f:%s' % (t, '-'.join(map(str, c))) for t, c in res[:5]), flush=True)


def _dense2c_run(x, state, layer, n, cfg):
    nw, ntb, s, ks = cfg
    m = x.shape[0]
    nch = 1 if m <= 8 else 2 if m <= 16 else 4
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    tiles, emax, eoff, esc = state['c_' + layer]
    nt = n // 16
    state['dense_mod'].dense2c_gemm(nw, ntb, nch, s, ks, tiles[:nt], x, out, emax, eoff[:nt + 1], esc)
    return out


_dense_lab_plain = _dense_lab


def _dense_lab(state):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    shapes = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
    for layer, (n_full, k) in shapes.items():
        for m in (8, 16, 32):
            n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
            x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
            base = state[layer][:n]
            ref = x.float() @ base.float().T
            cub = _dlab_time(lambda: x @ base.T, flush)
            for name, runner in (('P', _dense2_run), ('C', _dense2c_run)):
                res = []
                bad = 0
                for nw, ntb in ((8, 1), (4, 1), (4, 2)):
                    for s in (4, 6):
                        for ks in (2, 4, 8):
                            cfg = (nw, ntb, s, ks)
                            try:
                                o = runner(x, state, layer, n, cfg).float()
                                torch.cuda.synchronize()
                            except Exception as ex:
                                continue
                            err = ((o - ref).norm() / ref.norm()).item()
                            if err > 5e-3:
                                bad += 1
                                continue
                            res.append((_dlab_time(lambda: runner(x, state, layer, n, cfg), flush, 8), cfg))
                res.sort()
                print('LABE2', layer, 'm=%d' % m, name, 'cublas=%.1f' % cub, 'bad=%d' % bad, ' '.join('%.1f:%s' % (t, '-'.join(map(str, c))) for t, c in res[:4]), flush=True)
