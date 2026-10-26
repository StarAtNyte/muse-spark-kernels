

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


def _probe_lab(state):
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    mod = state['dense_mod']
    out = torch.zeros(4, dtype=torch.float32, device='cuda')
    for mb in (8, 16, 38, 67, 256):
        buf = torch.randint(0, 255, (mb * 1024 * 1024,), dtype=torch.uint8, device='cuda')
        res = []
        for blocks, threads in ((132 * 4, 256), (132 * 8, 256), (132 * 16, 256), (132 * 4, 512), (132 * 2, 1024), (132 * 32, 128)):
            t = _dlab_time(lambda: mod.probe_t(buf, out, blocks, threads), flush)
            res.append('%dx%d=%.1f' % (blocks, threads, t))
        print('LABP read %dMB' % mb, ' '.join(res), flush=True)
        del buf


def _dense_lab(state):
    _probe_lab(state)
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    shapes = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096), 'router': (256, 8192)}
    for layer, (n_full, k) in shapes.items():
        for m in (1, 8, 16, 32, 128, 256):
            n = 1024 if (layer == 'unpooler' and m <= 8) else n_full
            x = torch.randn((m, k), device='cuda').to(torch.bfloat16)
            data = {'x': x, 'config': {'layer': layer, 'n': n, 'm': m, 'k': k}}
            try:
                old = _dlab_time(lambda: _old_kernel_fn(data, state), flush)
            except Exception:
                old = -1
            base = (state[layer][:n] if layer != 'router' else None)
            cub = _dlab_time(lambda: x @ base.T, flush) if base is not None else -1
            kcs = k // 64
            ntot = 512 if layer == 'router' else n
            res = []
            for ntb in (1, 2):
                if (ntot // 16) % ntb:
                    continue
                for nw in (2, 4, 8, 16):
                    for ksplit, fix in ((1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1)):
                        if kcs % (ksplit * nw):
                            continue
                        for nch in ((1,) if m <= 8 else (1, 2, 4) if m <= 32 else (2, 4, 8)):
                            if nch == 8 and (ntb != 1 or nw == 2 or nw == 16):
                                continue
                            if nw == 16 and (ntb == 2 and nch != 1):
                                continue
                            cfg = (ntb, nch, nw, 4, ksplit, fix)
                            try:
                                t = _dlab_time(lambda: _dense_run(x, state, layer, n, cfg), flush)
                            except Exception:
                                continue
                            res.append((t, cfg))
            res.sort()
            chk = 'na'
            if res:
                try:
                    o = _dense_run(x, state, layer, n, res[0][1]).float()
                    w = state[layer][:n].float() if layer != 'router' else state['router'].float()
                    ref = x.float() @ w.T
                    chk = '%.1e' % ((o - ref).norm() / ref.norm()).item()
                except Exception as ex:
                    chk = 'ERR'
            pd = ''
            if res:
                for pv in (0, 1):
                    state['dense_mod'].dense_set_pdl_t(pv)
                    pd += ' pdl%d=%.1f' % (pv, _dlab_time(lambda: _dense_run(x, state, layer, n, res[0][1]), flush))
                state['dense_mod'].dense_set_pdl_t(int(_DENSE_PDL and torch.cuda.get_device_capability()[0] >= 9))
                for pv in (0, 1):
                    state['dense_mod'].dense_set_pf_t(pv)
                    pd += ' pf%d=%.1f' % (pv, _dlab_time(lambda: _dense_run(x, state, layer, n, res[0][1]), flush))
                state['dense_mod'].dense_set_pf_t(0)
            print('LABD', layer, 'm=%d' % m, 'chk=' + chk, 'cublas=%.1f' % cub, 'v5=%.1f' % old, ' '.join('%.1f:%s' % (t, '-'.join(map(str, c))) for t, c in res[:5]) + pd, flush=True)
