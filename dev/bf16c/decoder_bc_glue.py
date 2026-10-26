

_BC_MOD = None


def _bc_module():
    global _BC_MOD
    if _BC_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _BC_MOD = load_inline("muse_bf16c", _BC_CPP, cuda_sources=_BC_CUDA, functions=["bf16c"], extra_cuda_cflags=["-O3"], verbose=False)
    return _BC_MOD


_BC_CFGS = [(1, 8, 4), (1, 16, 4), (1, 16, 6), (1, 32, 3), (1, 32, 4), (2, 8, 4), (2, 16, 4), (2, 16, 6), (1, 8, 8)]


def _bc_bucket(m):
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 0


def _bc_run(state, layer, x, n, cfg):
    pk = state["bc"][layer]
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    _bc_module().bf16c(pk["wc"], x, out, pk["esc_off"], pk["esc"], n, x.shape[1], pk["base"], cfg[0], cfg[1], cfg[2])
    return out


def _bc_time(fn, flush, reps=10):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.sum()
        torch.cuda.synchronize()
        s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_.record()
        g.replay()
        e_.record()
        torch.cuda.synchronize()
        ts.append(s_.elapsed_time(e_) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _bc_tune(state, weights):
    state["bc"] = {}
    state["bc_pick"] = {}
    try:
        _bc_module()
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            wc, esc_off, esc, base, _st = _bc_pack(weights[layer])
            state["bc"][layer] = dict(wc=wc, esc_off=esc_off, esc=esc, base=base)
        flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        ns = {"qkv_gate": 2304, "o": HIDDEN, "pooler": D_EXPERT}
        for layer in ("qkv_gate", "o", "pooler", "unpooler"):
            k = weights[layer].shape[1]
            for bucket in (1, 8, 16, 32):
                n = (UNPOOL_SLICE if bucket <= 8 else HIDDEN) if layer == "unpooler" else ns[layer]
                x = torch.randn((bucket, k), device="cuda").to(torch.bfloat16)
                ref = x.float() @ weights[layer][:n].float().T
                t_base = _bc_time(lambda: _dgemm(x, layer, n, state), flush)
                best = (t_base * 0.96, None)
                for cfg in _BC_CFGS:
                    if (k // 64) % cfg[1] != 0 or n % (16 * cfg[0]) != 0:
                        continue
                    try:
                        o = _bc_run(state, layer, x, n, cfg)
                        torch.cuda.synchronize()
                        if ((o.float() - ref).norm() / ref.norm()).item() > 3e-3:
                            continue
                        t = _bc_time(lambda: _bc_run(state, layer, x, n, cfg), flush)
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, cfg)
                state["bc_pick"][(layer, bucket)] = best[1]
                print("BCPICK", layer, bucket, best[1], "base=%.1f best=%.1f" % (t_base, best[0]), flush=True)
    except Exception as ex:
        print("BC disabled:", repr(ex)[:200], flush=True)
        state["bc_pick"] = {}


_prepare_nobc = prepare


def prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    fns = [f for f in (globals().get("_wm_module"), globals().get("_fd_module"), globals().get("_dense_module"),
                       globals().get("_wg_module"), globals().get("_bc_module")) if f]
    with ThreadPoolExecutor(len(fns)) as ex:
        list(ex.map(lambda f: f(), fns))
    state = _prepare_nobc(weights)
    _bc_tune(state, weights)
    return state
