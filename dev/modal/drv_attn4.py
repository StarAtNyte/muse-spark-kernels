import sys; sys.path.insert(0, "/w/kernels/muse_attention_decode")
exec(open('/w/modal/drv_attn3.py').read().split("LAYER = os.environ")[0].replace('__PROBLEM__', 'muse_attention_decode'))
LAYER = os.environ.get("LAYER", "full")
for B in [int(x) for x in os.environ.get("BATCHES", "1").split(",")]:
    d = R.generate_input(B, 80000, "tp8", LAYER, 3, q_len=8, spread="agentic" if B > 1 else "fixed", dist="coherent" if LAYER == "full" else "soft", page=1)
    res = []
    m._WG_MINB = 10**6
    res.append("v7path=%.1f" % timeit(lambda: m.kernel_fn(d)))
    base = m.kernel_fn(d).float().clone()
    m._WG_MINB = 1
    for sp in eval(os.environ.get("SPL", "[16,32,64,128,256]")):
        m._WG_SPLITS = sp
        try:
            o = m.kernel_fn(d).float()
            err = ((o - base).norm() / base.norm()).item()
            res.append("wg5/p%d=%.1f(e%.0e)" % (sp, timeit(lambda: m.kernel_fn(d)), err))
        except Exception as ex:
            res.append("wg5/p%d=ERR(%s)" % (sp, str(ex)[:40]))
    print("SWEEP verify", LAYER, "B=%d" % B, " ".join(res), flush=True)
