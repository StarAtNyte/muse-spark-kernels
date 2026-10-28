"""Screen the dual-phase designated WGS finisher for MoE M=32."""
exec(open("/w/modal/drv_moe_vjeux_nowgs.py").read().split('base_defines = tuple')[0])

base_defines = tuple(m.MAIN_DEFINES)
variant_defines = base_defines + ("@wgsdfin3",)
m.ALT_SRCS["wgsdfin3"] = m.CUDA_SRC.replace(
    "#define OKC_WGSDFIN 2", "#define OKC_WGSDFIN 3", 1)
assert "#define OKC_WGSDFIN 3" in m.ALT_SRCS["wgsdfin3"]
variant = m._ext(variant_defines + tuple(state["sfx"]))
print("READY_VARIANT", torch.cuda.get_device_name(), "sfx", state["sfx"], flush=True)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)


def capture(data, defines):
    m.MAIN_DEFINES = defines
    out = m.kernel_fn(data, state)
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, out, weights)
    if not ok:
        raise AssertionError(("reference", defines, msg))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m.kernel_fn(data, state)
    graph.replay(); torch.cuda.synchronize()
    return graph, out


def elapsed(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000


for seed in (419, 7919):
    data = R.generate_input(32, "ep8", seed)
    gb, ob = capture(data, base_defines)
    gc, oc = capture(data, variant_defines)
    err = ((ob.float() - oc.float()).norm() / ob.float().norm().clamp_min(1e-20)).item()
    bt, ct, ratios = [], [], []
    for i in range(24):
        if i & 1:
            c, b = elapsed(gc), elapsed(gb)
        else:
            b, c = elapsed(gb), elapsed(gc)
        bt.append(b); ct.append(c); ratios.append(b / c)
    print("RESULT", json.dumps({"seed": seed, "base_us": statistics.median(bt),
          "variant_us": statistics.median(ct), "base_over_variant": statistics.median(ratios),
          "relative_l2": err}), flush=True)
    del data, gb, gc, ob, oc
print("MOE_WGSDFIN3_SCREEN_DONE", flush=True)
