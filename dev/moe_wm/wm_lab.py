

def _lab_inputs(m, device, seed):
    """Synthetic routing resembling the judge's busiest-GPU normal distribution; experts 0-31 local."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((m, 256), generator=g, device=device)
    z = torch.randn((256,), generator=g, device=device)
    u = torch.rand((256,), generator=g, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": m, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def _lab_time(fn, flush, reps=20):
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


_WM_EXTRA = ['-DWM_DBG_BUILD']


def _lab_fin(state):
    global _WM_FIN, _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    saved = _WM_CFG
    for m, cfg in ((1, (2, 4, 0)), (16, (2, 4, 0)), (32, (4, 4, 1)), (256, (4, 4, 1))):
        data = _lab_inputs(m, 'cuda', 100 + m)
        _WM_CFG = [(10**9, cfg)]
        row = [f'FIN M={m}']
        for fv in (False, True, False, True):
            _WM_FIN = fv
            row.append(f'fin{int(fv)}=' + format(_lab_time(lambda: kernel_fn(data, state), flush), '.1f'))
        print('LAB', ' '.join(row), flush=True)
    _WM_FIN = False
    _WM_CFG = saved


def _lab_check(state):
    global _WM_CFG, _WM_FIN
    saved = _WM_CFG
    for m in (1, 8, 16, 32, 128, 256):
        data = _lab_inputs(m, 'cuda', 300 + m)
        ref = _marlin_kernel_fn(data, state).float()
        for cfg in ((2, 4, 0), (2, 4, 2), (4, 4, 2), (4, 4, 1)):
            if m > 16 and cfg[2] == 0:
                continue
            for fin in (False, True):
                _WM_CFG = [(10**9, cfg)]
                _WM_FIN = fin
                try:
                    out = kernel_fn(data, state).float()
                    err = ((out - ref).norm() / ref.norm()).item()
                    print('LABC M=%d cfg=%s fin=%d relerr_vs_marlin=%.2e' % (m, cfg, int(fin), err), flush=True)
                except Exception as ex:
                    print('LABC M=%d cfg=%s fin=%d ERR %s' % (m, cfg, int(fin), str(ex)[:60]), flush=True)
    _WM_FIN = False
    _WM_CFG = saved


def _lab_stress(state):
    global _WM_CFG, _WM_FIN
    saved = _WM_CFG
    flush = torch.zeros(64 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    for m, cfg in ((32, (4, 4, 1)), (32, (2, 4, 1)), (16, (2, 4, 0)), (32, (4, 4, 0))):
        for pdl in (0, 1):
            _wm_module().wm_set_pdl_t(pdl)
            _WM_CFG = [(10**9, cfg)]
            bad = {'eager': 0, 'graph': 0}
            maxerr = 0.0
            n_trials = 120
            for trial in range(n_trials):
                data = _lab_inputs(m, 'cuda', 5000 + 7 * trial)
                try:
                    ref = _marlin_kernel_fn(data, state).float()
                    out = kernel_fn(data, state).float()
                    err = ((out - ref).norm() / ref.norm()).item()
                    maxerr = max(maxerr, err)
                    if err > 1.0e-2:
                        bad['eager'] += 1
                    if trial % 4 == 0:
                        g = torch.cuda.CUDAGraph()
                        kernel_fn(data, state)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(g):
                            o2 = kernel_fn(data, state)
                        for _ in range(3):
                            flush.add_(1)
                            g.replay()
                        torch.cuda.synchronize()
                        err2 = ((o2.float() - ref).norm() / ref.norm()).item()
                        maxerr = max(maxerr, err2)
                        if err2 > 1.0e-2:
                            bad['graph'] += 1
                except Exception as ex:
                    print('LABS ERR', m, cfg, str(ex)[:80], flush=True)
                    break
            print('LABS M=%d cfg=%s pdl=%d bad_eager=%d bad_graph=%d maxerr=%.2e of %d' % (m, cfg, pdl, bad['eager'], bad['graph'], maxerr, n_trials), flush=True)
    _wm_module().wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    _WM_CFG = saved


def _lab(state):
    global _WM_DBG
    _lab_run(state)


def _lab_pdl(state):
    global _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    saved = _WM_CFG
    for m, cfg in ((1, (2, 4, 0)), (8, (2, 4, 0)), (16, (2, 4, 0)), (32, (4, 4, 1)), (128, (4, 4, 1)), (256, (2, 4, 1))):
        data = _lab_inputs(m, 'cuda', 100 + m)
        _WM_CFG = [(10**9, cfg)]
        row = [f'PDL M={m} cfg={cfg}']
        for pdl in (0, 1, 0, 1):
            _wm_module().wm_set_pdl_t(pdl)
            row.append(f'p{pdl}=' + format(_lab_time(lambda: kernel_fn(data, state), flush), '.1f'))
        print('LAB', ' '.join(row), flush=True)
    _wm_module().wm_set_pdl_t(int(_WM_PDL and torch.cuda.get_device_capability()[0] >= 9))
    _WM_CFG = saved


def _lab_dbg(state):
    global _WM_CFG, _WM_DBG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    saved = _WM_CFG
    for m, cfg in ((64, (4, 4, 4)), (128, (4, 4, 4)), (256, (4, 4, 4))):
        data = _lab_inputs(m, 'cuda', 100 + m)
        _WM_CFG = [(10**9, cfg)]
        row = [f'DBG M={m} cfg={cfg}']
        for d in (0, 32):
            _WM_DBG = d
            try:
                row.append(f'd{d}=' + format(_lab_time(lambda: kernel_fn(data, state), flush), '.1f'))
            except Exception as ex:
                row.append(f'd{d}=ERR')
        print('LAB', ' '.join(row), flush=True)
    _WM_DBG = 0
    _WM_CFG = saved


def _lab_cfgs(state):
    global _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    saved = _WM_CFG
    lines = []
    for m in (64, 128, 256, 1024):
        data = _lab_inputs(m, "cuda", 100 + m)
        nact = len(torch.unique(data["topk_ids"][data["topk_ids"] < 32]))
        base = 0.0
        row = [f"M={m} act={nact} marlin={base:.1f}"]
        for cfg in _LAB_CFGS:
            if m > 16 and cfg[2] == 0 and True:
                nmax_ok = False
            else:
                nmax_ok = True
            if not nmax_ok:
                continue
            _WM_CFG = [(10**9, cfg)]
            try:
                t = _lab_time(lambda: kernel_fn(data, state), flush)
                row.append(f"{cfg[0]}{cfg[1]}{cfg[2]}={t:.1f}")
            except Exception as ex:  # unsupported combination
                row.append(f"{cfg}=ERR")
        lines.append(" ".join(row))
        print("LAB", lines[-1], flush=True)
    _WM_CFG = saved


def _lab_persist(state):
    global _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    saved = _WM_CFG
    for m, cfg in ((1, (2, 4, 0)), (8, (2, 4, 0)), (16, (2, 4, 0)), (32, (4, 4, 1)), (64, (4, 4, 1)), (128, (4, 4, 1))):
        data = _lab_inputs(m, 'cuda', 100 + m)
        _WM_CFG = [(10**9, cfg)]
        row = [f'PERS M={m} cfg={cfg}']
        for pv in (0, 1, 2, 3, 4, 0, 1, 2, 3, 4):
            _wm_module().wm_set_persist_t(pv)
            row.append(f'k{pv}=' + format(_lab_time(lambda: kernel_fn(data, state), flush), '.1f'))
        print('LAB', ' '.join(row), flush=True)
    _wm_module().wm_set_persist_t(0)
    _WM_CFG = saved


def _lab_prof(state):
    global _WM_CFG, _WM_DBG
    from torch.profiler import profile, ProfilerActivity
    for pdl in (0, 1):
        _wm_module().wm_set_pdl_t(pdl)
        for m, cfg in ((1, (2, 4, 0)), (8, (2, 4, 0)), (32, (4, 4, 1)), (128, (4, 4, 4)), (256, (4, 4, 4))):
            data = _lab_inputs(m, 'cuda', 100 + m)
            _WM_CFG = [(10**9, cfg)]
            for _ in range(3):
                kernel_fn(data, state)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    kernel_fn(data, state)
                torch.cuda.synchronize()
            for ev in prof.key_averages():
                if ev.device_time_total > 0:
                    print('LABP pdl=%d M=%d %s n=%d avg=%.1fus' % (pdl, m, ev.key[:40], ev.count, ev.device_time_total / max(ev.count, 1)), flush=True)


def _lab_run(state):
    global _WM_CFG
    data = _lab_inputs(256, 'cuda', 356)
    _WM_CFG = [(10**9, (4, 4, 4))]
    for _ in range(3):
        kernel_fn(data, state)
    torch.cuda.synchronize()
