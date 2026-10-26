

_WMX_SRC = __WMX__


def _dbg_lab(state, weights):
    import types
    wmx = types.ModuleType("wmx_lab")
    exec(compile(_WMX_SRC, "wmx_lab", "exec"), wmx.__dict__)
    dev = weights["w13"].device
    wm = wmx.build_wm_experts(weights["w13"][:32].contiguous(), weights["w13_sf"][:32].contiguous(), weights["w13_gs"][:32].contiguous(),
                              weights["w2"][:32].contiguous(), weights["w2_sf"][:32].contiguous(), weights["w2_gs"][:32].contiguous())
    mod = wmx._get_module()
    for m, nch in ((8, 4), (40, 8), (100, 16), (753, 16)):
        g = torch.Generator(device=dev).manual_seed(5 + m)
        x = torch.randn((m, 4096), generator=g, device=dev).to(torch.bfloat16)
        ids = torch.randint(0, 32, (m, 8), generator=g, device=dev).to(torch.int32)
        lmax = (m + 7) // 8 * 8
        cnt = torch.empty(32, dtype=torch.int32, device=dev)
        lst = torch.empty(32 * lmax, dtype=torch.int32, device=dev)
        act = torch.empty(4 * (32 + (m * 8) // (nch * 8) + 8), dtype=torch.int32, device=dev)
        woff = torch.empty(128, dtype=torch.int32, device=dev)
        nact = torch.empty(1, dtype=torch.int32, device=dev)
        lids = torch.empty(m * 8, dtype=torch.int32, device=dev)
        empty = torch.empty(0, dtype=torch.int32, device=dev)
        try:
            mod.wm_route(ids, empty, 0, lmax, cnt, lst, act, nact, lids, woff, nch * 8)
            torch.cuda.synchronize(); print("LABD route ok m=%d" % m, flush=True)
            xp = torch.empty_like(x)
            mod.wm_permx(x, xp)
            torch.cuda.synchronize(); print("LABD permx ok", flush=True)
            maxslots = min(m * 8, 32 + (m * 8) // (nch * 8))
            inter = torch.zeros((m * 8, 4096), dtype=torch.bfloat16, device=dev)
            y = torch.zeros((m * 8, 4096), dtype=torch.bfloat16, device=dev)
            mod.wm_gemm_run(0, nch, 3, 4, 7, 0, wm.w13, wm.s13, wm.gs13, xp, inter, cnt, lst, act, nact, woff, maxslots, lmax)
            torch.cuda.synchronize(); print("LABD gemm0 ok", flush=True)
            mod.wm_gemm_run(1, nch, 3, 4, 7, 0, wm.w2, wm.s2, wm.gs2, inter, y, cnt, lst, act, nact, woff, maxslots, lmax)
            torch.cuda.synchronize(); print("LABD gemm1 ok", flush=True)
            # reference: the validated row-split path
            nch0 = 4
            act0 = torch.empty(4 * (32 + (m * 8) // 32 + 8), dtype=torch.int32, device=dev)
            mod.wm_route(ids, empty, 0, lmax, cnt, lst, act0, nact, lids, woff, 32)
            ms0 = min(m * 8, 32 + (m * 8) // 32)
            inter0 = torch.zeros((m * 8, 4096), dtype=torch.bfloat16, device=dev)
            y0 = torch.zeros((m * 8, 4096), dtype=torch.bfloat16, device=dev)
            mod.wm_gemm_run(0, 4, 4, 4, 4, 0, wm.w13, wm.s13, wm.gs13, x, inter0, cnt, lst, act0, nact, woff, ms0, lmax)
            mod.wm_gemm_run(1, 4, 4, 4, 4, 0, wm.w2, wm.s2, wm.gs2, inter0, y0, cnt, lst, act0, nact, woff, ms0, lmax)
            torch.cuda.synchronize()
            live = (lids >= 0)
            e1 = ((y.float() - y0.float())[live].norm() / y0.float()[live].norm()).item()
            print("LABD m=%d y relerr vs rs path %.3e" % (m, e1), flush=True)
        except Exception as ex:
            print("LABD ERR m=%d %s" % (m, str(ex)[:150]), flush=True)
            break
