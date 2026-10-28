

def _lab_inputs(m, device, seed):
    """Synthetic routing resembling the judge's busiest-GPU normal distribution; experts 0-31 local."""
    try:
        import reference as _R
        d = _R.generate_input(m, "ep8", seed, device=device)
        return d
    except Exception:
        pass
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


_WM_EXTRA = ['-DWM_DBG_BUILD', '-gencode=arch=compute_90a,code=sm_90a']


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
    import os
    (_lab_run if os.environ.get('LABMODE') == 'run' else _lab_wgcmp)(state)


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
    for m, cfg in ((32, (4, 4, 1)),):
        datas = [_lab_inputs(m, 'cuda', 100 + 7 * m + s) for s in range(3)]
        _WM_CFG = [(10**9, cfg)]
        row = [f'DBG M={m} cfg={cfg}']
        for d in (0, 1, 2, 4, 8, 3, 11, 9):
            _WM_DBG = d
            try:
                ts = [_lab_time(lambda: kernel_fn(dd, state), flush) for dd in datas]
                row.append(f'd{d}=' + '/'.join(f'{t:.0f}' for t in ts))
            except Exception as ex:
                row.append(f'd{d}=ERR')
        print('LAB', ' '.join(row), flush=True)
    _WM_DBG = 0
    _WM_CFG = saved


def _lab_cfgs(state):
    global _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    saved = _WM_CFG
    for m in _LAB_MS:
        datas = [_lab_inputs(m, "cuda", 100 + 7 * m + s) for s in range(_LAB_SEEDS)]
        nact = [len(torch.unique(d["topk_ids"][d["topk_ids"] < 32])) for d in datas]
        row = [f"M={m} act={nact}"]
        for cfg in _LAB_CFGS:
            if m > 16 and cfg[2] == 0:
                continue
            _WM_CFG = [(10**9, cfg)]
            try:
                ts = [_lab_time(lambda: kernel_fn(d, state), flush) for d in datas]
                row.append(f"{cfg[0]}{cfg[1]}{cfg[2]}=" + "/".join(f"{t:.0f}" for t in ts) + f"(avg {sum(ts)/len(ts):.1f})")
            except Exception as ex:
                row.append(f"{cfg}=ERR")
        print("LAB", " ".join(row), flush=True)
    _WM_CFG = saved


_LAB_MS = (8, 16, 32)
_LAB_SEEDS = 4


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
    import os
    m = int(os.environ.get("LABM", "256"))
    cfg = eval(os.environ.get("LABCFG", "(4, 4, 4)"))
    data = _lab_inputs(m, 'cuda', 100 + 7 * m)
    _WM_CFG = [(10**9, cfg)]
    for _ in range(3):
        kernel_fn(data, state)
    torch.cuda.synchronize()


def _lab_pf(state):
    global _WM_CFG, _WM_PF
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    cfgs = {1: (2, 4, 0), 8: (2, 2, 0), 16: (2, 4, 0), 32: (4, 4, 1), 128: (4, 4, 4), 256: (4, 4, 4)}
    for m in (1, 8, 16, 32, 128, 256):
        datas = [_lab_inputs(m, 'cuda', 100 + 7 * m + s) for s in range(4)]
        _WM_CFG = [(10**9, cfgs[m])]
        row = [f'PF M={m}']
        for npf in (0, 1, 2, 3, 4, 6, 8):
            _WM_PF = npf
            try:
                ts = [_lab_time(lambda: kernel_fn(dd, state), flush) for dd in datas]
                row.append(f'pf{npf}=' + '/'.join(f'{t:.0f}' for t in ts) + f'({sum(ts)/len(ts):.1f})')
            except Exception as ex:
                row.append(f'pf{npf}=ERR')
        print('LAB', ' '.join(row), flush=True)
    _WM_PF = 0


def _lab_chain(state):
    """Marginal cost of each kernel in the MoE chain (graph + cold L2): route, +gemm0, +gemm1, +finish."""
    global _WM_CFG
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    cfgs = {1: (2, 4, 0), 8: (2, 2, 0), 16: (2, 8, 0), 32: (4, 4, 1), 128: (4, 4, 4), 256: (4, 4, 4)}
    for m in (1, 8, 16, 32, 128, 256):
        datas = [_lab_inputs(m, 'cuda', 100 + 7 * m + s) for s in range(3)]
        _WM_CFG = [(10**9, cfgs[m])]
        wm = state["wm"]
        mod = wm["mod"]
        row = [f'CHAIN M={m}']
        for stage in (1, 2, 3, 4):
            def run(d, stage=stage):
                x = d["hidden_states"].contiguous()
                nch = 1 if m <= 8 else 2 if m <= 16 else 4
                lmax = (m + 7) // 8 * 8
                _c = _wm_cfg(m)
                nt, nw, ring = _c[:3]
                dev = x.device
                ids = d["topk_ids"].contiguous()
                cnt = torch.empty(32, dtype=torch.int32, device=dev)
                lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
                act = torch.empty(512, dtype=torch.int32, device=dev)
                woff = torch.empty(128, dtype=torch.int32, device=dev)
                nact = torch.empty(1, dtype=torch.int32, device=dev)
                lids = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
                mod.wm_route(ids, _WM_NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
                if stage == 1:
                    return act
                maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
                inter = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
                y = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)
                mod.wm_gemm_run(0, nch, nt, nw, ring, 0, wm["w13"], wm["s13"], wm["gs13"], x, inter, cnt, lst, act, nact, woff, maxslots, lmax)
                if stage == 2:
                    return inter
                mod.wm_gemm_run(1, nch, nt, nw, ring, 0, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
                if stage == 3:
                    return y
                tw = d["topk_weights"].to(torch.float32).contiguous()
                out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
                mod.wm_finish(y, tw, lids, state["norm_w"], out, POST_NORM_EPS)
                return out
            try:
                ts = [_lab_time(lambda: run(dd), flush) for dd in datas]
                row.append(f's{stage}=' + '/'.join(f'{t:.0f}' for t in ts) + f'({sum(ts)/len(ts):.1f})')
            except Exception as ex:
                row.append(f's{stage}=ERR({str(ex)[:40]})')
        print('LAB', ' '.join(row), flush=True)


def _lab_wgcmp(state):
    global _WM_CFG, _WM_FIN
    _WM_FIN = True
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
    for m in (128, 256):
        datas = [_lab_inputs(m, 'cuda', 100 + 7 * m + s) for s in range(3)]
        for d in datas[:1]:
            _WM_CFG = [(10**9, (4, 4, 4))]
            a = kernel_fn(d, state).float()
            row = [f'WGCMP M={m}']
            for cfg in ((3, 4, 7), (2, 4, 7)):
                _WM_CFG = [(10**9, cfg)]
                try:
                    b = kernel_fn(d, state).float()
                    row.append(f'{cfg}: relerr_vs_rs={((a-b).norm()/a.norm()).item():.2e} finite={int(torch.isfinite(b).all())}')
                except Exception as ex:
                    row.append(f'{cfg}: ERR {str(ex)[:50]}')
            print('LAB', ' '.join(row), flush=True)
        for cfg in ((4, 4, 4), (3, 4, 7), (2, 4, 7)):
            _WM_CFG = [(10**9, cfg)]
            try:
                ts = [_lab_time(lambda: kernel_fn(dd, state), flush) for dd in datas]
                print('LAB', f'WGT M={m} {cfg}', '/'.join(f'{t:.0f}' for t in ts), flush=True)
            except Exception as ex:
                print('LAB', f'WGT M={m} {cfg} ERR {str(ex)[:50]}', flush=True)


def _lab_wgdbg(state):
    global _WM_CFG
    wm = state["wm"]; mod = wm["mod"]
    m = 256
    d = _lab_inputs(m, 'cuda', 100 + 7 * m)
    x = d["hidden_states"].contiguous()
    ids = d["topk_ids"].contiguous()
    k = torch.arange(4096, device='cuda')
    c, kl = k >> 6, k & 63
    t_, ks, p_, h_ = kl >> 4, (kl >> 2) & 3, (kl >> 1) & 1, kl & 1
    perm = c * 64 + (ks * 2 + p_) * 8 + t_ * 2 + h_
    def run(cfg, permx):
        nch = 4
        lmax = (m + 7) // 8 * 8
        nt, nw, ring = cfg
        cnt = torch.empty(32, dtype=torch.int32, device='cuda'); lst = torch.empty(32 * lmax, dtype=torch.int32, device='cuda')
        act = torch.empty(512, dtype=torch.int32, device='cuda'); woff = torch.empty(128, dtype=torch.int32, device='cuda')
        nact = torch.empty(1, dtype=torch.int32, device='cuda'); lids = torch.empty(m * TOP_K, dtype=torch.int32, device='cuda')
        mod.wm_route(ids, _WM_NOLS, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
        maxslots = min(m * TOP_K, 32 + (m * TOP_K) // (nch * 8))
        xx = x
        if permx:
            xx = torch.empty_like(x); mod.wm_permx(x, xx)
        inter = torch.zeros((m * TOP_K, 4096), dtype=torch.bfloat16, device='cuda')
        y = torch.zeros((m * TOP_K, 4096), dtype=torch.bfloat16, device='cuda')
        mod.wm_gemm_run(0, nch, nt, nw, ring, 0, wm["w13"], wm["s13"], wm["gs13"], xx, inter, cnt, lst, act, nact, woff, maxslots, lmax)
        mod.wm_gemm_run(1, nch, nt, nw, ring, 0, wm["w2"], wm["s2"], wm["gs2"], inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
        torch.cuda.synchronize()
        return inter, y, lids
    i1, y1, l1 = run((4, 4, 4), False)
    i2, y2, l2 = run((3, 4, 7), True)
    live = (l1.view(-1) >= 0)
    a = i1[live].float(); b = i2[live][:, perm].float()
    print('LAB WGDBG inter relerr(un-permuted)=%.3e  nonzero_rs=%d nonzero_wg=%d live=%d' % (((a - b).norm() / a.norm()).item(), int((a != 0).sum()), int((b != 0).sum()), int(live.sum())), flush=True)
    a = y1[live].float(); b = y2[live].float()
    print('LAB WGDBG y relerr=%.3e' % (((a - b).norm() / a.norm()).item()), flush=True)
