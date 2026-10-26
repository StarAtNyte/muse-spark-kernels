

# ---- lossless 12-bit BF16 weights + swap-AB mma.sync GEMV for 1..32 tokens (own CUDA), chosen per (layer, m) by timing in prepare() ----
_BC_MOD = None


def _bc_module():
    global _BC_MOD
    if _BC_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _BC_MOD = load_inline("muse_bf16c", _BC_CPP, cuda_sources=_BC_CUDA, functions=["bf16c"], extra_cuda_cflags=["-O3"], verbose=False)
    return _BC_MOD


_BC_CFGS = [(1, 8, 4), (1, 8, 8), (1, 16, 4), (1, 16, 6), (1, 16, 8), (1, 32, 3), (1, 32, 4), (2, 8, 4), (2, 16, 4), (2, 16, 6), (1, 4, 4), (2, 4, 4)]
_BC_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')


def _bc_bucket(m):
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 0


def _bc_run(state, layer, x, n, cfg):
    pk = state['bc'][layer]
    k = x.shape[1]
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    _bc_module().bf16c(pk['wc'], x, out, pk['esc_off'], pk['esc'], n, k, pk['base'], cfg[0], cfg[1], cfg[2])
    return out


def _bc_time(fn, flush, reps=12):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.sum()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def prepare(weights):
    state = _prepare_v5(weights)
    state['bc'] = {}
    state['bc_pick'] = {}
    try:
        mod = _bc_module()
        for layer in _BC_LAYERS:
            wc, esc_off, esc, base, _stats = _bc_pack(weights[layer])
            state['bc'][layer] = dict(wc=wc, esc_off=esc_off, esc=esc, base=base)
        flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
        for layer in _BC_LAYERS:
            n_full, k = weights[layer].shape
            for bucket in (1, 8, 16, 32):
                n = 1024 if (layer == 'unpooler' and bucket <= 8) else n_full
                if layer == 'unpooler' and bucket <= 8 and False:
                    continue
                x = torch.randn((bucket, k), device='cuda').to(torch.bfloat16)
                base_fn = lambda: _kernel_v5({'x': x, 'config': {'layer': layer, 'n': n}}, state)
                ref = x.float() @ weights[layer][:n].float().T
                t_base = _bc_time(base_fn, flush)
                best = (t_base * 0.97, None)
                for cfg in _BC_CFGS:
                    if (k // 64) % cfg[1] != 0 or n % (16 * cfg[0]) != 0:
                        continue
                    try:
                        o = _bc_run(state, layer, x, n, cfg)
                        torch.cuda.synchronize()
                        err = ((o.float() - ref).norm() / ref.norm()).item()
                        if err > 3e-3:
                            continue
                        t = _bc_time(lambda: _bc_run(state, layer, x, n, cfg), flush)
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, cfg)
                state['bc_pick'][(layer, bucket)] = best[1]
                print('BCPICK', layer, bucket, best[1], 'base=%.1f best=%.1f' % (t_base, best[0]), flush=True)
    except Exception as ex:  # any failure: stock v5 paths only
        print('BC disabled:', repr(ex)[:200], flush=True)
        state['bc_pick'] = {}
    return state


def kernel_fn(data, weights):
    x = data['x']
    layer = data['config']['layer']
    if layer != 'router' and weights['bc_pick']:
        m = x.shape[0]
        b = _bc_bucket(m)
        cfg = weights['bc_pick'].get((layer, b)) if b else None
        if cfg is not None and x.dtype == torch.bfloat16 and x.is_contiguous():
            return _bc_run(weights, layer, x, data['config']['n'], cfg)
    return _kernel_v5(data, weights)
