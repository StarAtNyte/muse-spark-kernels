

_WMX_SRC = __WMX__


def _conv_lab(state, weights):
    import types
    wmx = types.ModuleType("wmx_lab")
    exec(compile(_WMX_SRC, "wmx_lab", "exec"), wmx.__dict__)
    dev = weights["w13"].device
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    t0 = time.time()
    wm = wmx.build_wm_experts(weights["w13"][:32].contiguous(), weights["w13_sf"][:32].contiguous(), weights["w13_gs"][:32].contiguous(),
                              weights["w2"][:32].contiguous(), weights["w2_sf"][:32].contiguous(), weights["w2_gs"][:32].contiguous())
    torch.cuda.synchronize()
    print("LABV module compile+build %.1fs" % (time.time() - t0), flush=True)
    wmx.reserve_scratch(dev)
    # exactness against the real Marlin tensors from the baseline repack
    bank = state["banks"][(0, 32)]
    w13, s13, g13, w2, s2, g2 = wmx.marlin_view(wm)
    torch.cuda.synchronize()
    print("LABV marlin w13 equal", bool(torch.equal(w13, bank["w13"])), "w2 equal", bool(torch.equal(w2, bank["w2"])),
          "s13 equal", bool(torch.equal(s13.view(torch.uint8), bank["w13_s"].view(torch.uint8))), "s2 equal", bool(torch.equal(s2.view(torch.uint8), bank["w2_s"].view(torch.uint8))),
          "gs13 equal", bool(torch.equal(g13, bank["w13_gs"])), "gs2 equal", bool(torch.equal(g2, bank["w2_gs"])), flush=True)
    t = _cv_time(lambda: wmx.marlin_view(wm), flush)
    print("LABV marlin_view (both matrices) us=%.1f" % t, flush=True)
    # wm path vs (conversion + stock Marlin) at growing batch sizes
    for m in (753, 2048, 4096, 8192, 16384):
        x, ids, tw = _lab_in(m, dev, 100 + m)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                "config": {"bs": m, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0, "local_count": 32, "top_k": 8, "sharding": "ep8"}}
        def wm_path():
            y, lids = wmx.expert_outputs(wm, x, ids, 0)
            return _finish_experts(y, tw, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=lids)
        ref = _marlin_kernel_fn(data, state).float() if m <= 8192 else None
        t_marlin = _cv_time(lambda: _marlin_kernel_fn(data, state), flush) if m <= 8192 else -1
        e0 = ((wm_path().float() - ref).norm() / ref.norm()).item() if ref is not None else 0.0
        row = ["err_vs_marlin=%.2e" % e0]
        saved = list(wmx._CFG)
        for name, cfg in (("t4n4", (4, 4, 4, 4)), ("wg", (3, 4, 7, 16)), ("wg2", (2, 4, 7, 16))):
            wmx._CFG[:] = [(10**9, cfg)]
            try:
                o2 = wm_path().float()
                e2 = ((o2 - ref).norm() / ref.norm()).item() if ref is not None else 0.0
                row.append("%s=%.1f(%.0e)" % (name, _cv_time(wm_path, flush), e2))
            except Exception as ex:
                row.append("%s=ERR" % name)
        wmx._CFG[:] = saved
        print("LABV m=%d marlin=%.1f %s" % (m, t_marlin, " ".join(row)), flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        wmx.marlin_view(wm)
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        if "wm_to_marlin" in ev.key or "raw_to_wm" in ev.key:
            print("LABV kernel", ev.key[:60], "us=%.1f" % ev.device_time_total, flush=True)


def _lab_in(m, device, seed):
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
    return x, ids.to(torch.int32), (routed.values / routed.values.sum(-1, keepdim=True)).float()


def _cv_time(fn, flush, reps=8):
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
