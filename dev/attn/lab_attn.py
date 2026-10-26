

def _lab_data(batch, q_len, layer, spread, ctx=80000, seed=0):
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(seed)
    if spread == 'agentic':
        u = torch.rand((batch,), generator=g, device=dev)
        seq_lens = (ctx * (1 + 0.3 * (2 * u - 1))).long().clamp(1, int(ctx * 1.3))
        longest = int(ctx * 1.3)
    else:
        seq_lens = torch.full((batch,), ctx, dtype=torch.int64, device=dev)
        longest = ctx
    max_pages = longest + q_len
    pages_per = seq_lens + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    offs = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offs[b])
        page_table[b, :n] = perm[o:o + n]
    shape = (total, 1, 2, 64)
    kc = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    vc = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    for c in (kc, vc):
        flat = c.view(-1, 128)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            flat[at:at + n] = torch.randn((n, 128), generator=g, device=dev).to(torch.float8_e4m3fn)
    hq, hkv = 16, 2
    qkv = torch.randn((batch, q_len, 2304), generator=g, device=dev).to(torch.bfloat16)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = torch.gather(page_table.long(), 1, pos).reshape(-1)
    return {'qkv': qkv, 'positions': positions, 'cache_loc': cache_loc, 'cache_seqlens': (seq_lens + q_len).to(torch.int32),
            'cu_seqlens_q': torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=dev),
            'scheduler_metadata': None, 'k_cache': kc, 'v_cache': vc,
            'k_scale': torch.tensor(1.0, device=dev), 'v_scale': torch.tensor(1.0, device=dev), 'page_table': page_table,
            'seq_lens': seq_lens.to(torch.int32),
            'config': {'batch': batch, 'q_len': q_len, 'num_q_heads': hq, 'num_kv_heads': hkv, 'head_dim': 64, 'page_size': 1,
                       'window': 2048 if layer == 'sliding' else 0, 'rope_theta': 500000.0 if layer == 'sliding' else 0.0,
                       'qk_scale_factor': 5.473, 'sm_scale': 0.125, 'kv_scale': 1.0}}


def _fd_check():
    # fused sliding decode vs v7
    for batch, ctx in ((2, 6000), (5, 3000), (1, 100)):
        data = _lab_data(batch, 1, 'sliding', 'agentic', ctx=ctx, seed=5 + batch)
        ref = _v7_kernel_fn(data).float()
        for st in (2, 3, 4):
            try:
                out = _slide_forward(data, st).float()
                print('LABAS B=%d ctx=%d stages=%d relerr_vs_v7=%.2e' % (batch, ctx, st, ((out - ref).norm() / ref.norm()).item()), flush=True)
            except Exception as ex:
                print('LABAS ERR', batch, st, str(ex)[:80], flush=True)
        del data
        torch.cuda.empty_cache()
    for batch, q_len, layer, ctx in ((2, 1, 'full', 20000), (3, 1, 'sliding', 6000), (2, 8, 'full', 12000), (2, 8, 'sliding', 6000)):
        data = _lab_data(batch, q_len, layer, 'agentic', ctx=ctx, seed=7 * batch + q_len)
        ref = _v7_kernel_fn(data).float()
        qkv, q = _fd_prepare(data)
        for variant, stages, splits, qk8 in ((1, 4, 8, 0), (1, 4, 8, 1), (2, 3, 16, 0), (2, 2, 8, 0)):
            if q_len > 1 and variant == 2:
                continue
            try:
                out = _fd_forward(data, q, qkv, stages, splits, qk8=qk8, variant=variant).float()
                err = ((out - ref).norm() / ref.norm()).item()
                print('LABAC B=%d L=%d %s variant=%d stages=%d splits=%d qk8=%d relerr_vs_v7=%.2e' % (batch, q_len, layer, variant, stages, splits, qk8, err), flush=True)
            except Exception as ex:
                print('LABAC', batch, q_len, layer, variant, 'ERR', str(ex)[:80], flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()


def _fd_lab():
    _fd_check()
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

    shapes = [(1, 1, 'sliding', 'fixed'), (1, 1, 'full', 'fixed'), (16, 1, 'sliding', 'agentic'), (16, 1, 'full', 'agentic'),
              (32, 1, 'sliding', 'agentic'), (32, 1, 'full', 'agentic'), (1, 8, 'sliding', 'fixed'), (1, 8, 'full', 'fixed'),
              (16, 8, 'sliding', 'agentic'), (16, 8, 'full', 'agentic'), (32, 8, 'sliding', 'agentic'), (32, 8, 'full', 'agentic')]
    for batch, q_len, layer, spread in shapes:
        data = _lab_data(batch, q_len, layer, spread, seed=batch * 10 + q_len)
        cfg = data['config']
        qkv, q = _fd_prepare(data)
        try:
            ref = timeit(lambda: _v7_kernel_fn(data))
        except Exception as ex:
            ref = -1
        res = []
        if layer == 'sliding' and q_len == 1:
            for st in (2, 3, 4):
                try:
                    res.append((timeit(lambda: _slide_forward(data, st)), st, 0, 99))
                except Exception as ex:
                    res.append((-1, st, 0, 99))
        grid_splits = (2, 4, 8, 16) if layer == 'sliding' else (8, 16, 32, 48, 64, 128)
        variants = [(2, 0, st) for st in (2, 3, 4)] if q_len == 1 else []
        variants += [(1, qk8, st) for qk8 in ((0,) if q_len > 1 else (0, 1)) for st in (3, 4)]
        for variant, qk8, stages in variants:
            for splits in grid_splits:
                try:
                    t = timeit(lambda: _fd_forward(data, q, qkv, stages, splits, qk8=qk8, variant=variant))
                    res.append((t, stages, splits, qk8 + 10 * variant))
                except Exception as ex:
                    res.append((-1, stages, splits, qk8 + 10 * variant))
        good = sorted(r for r in res if r[0] > 0)
        print('LABA', 'B=%d L=%d %s' % (batch, q_len, layer), 'v7=%.1f' % ref, ' '.join('%.1f:s%d/p%d/v%d' % r for r in good[:7]), flush=True)
        del data, qkv, q
        torch.cuda.empty_cache()
