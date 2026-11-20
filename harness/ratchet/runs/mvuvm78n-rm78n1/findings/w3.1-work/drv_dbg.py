
# ---------------- timeline driver ----------------
import os, sys, json, statistics, torch
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
CFGS = [("qkv_gate",128,(2,128,6,0,7,1,0,1)), ("qkv_gate",128,(2,128,6,0,4,1,1,0)),
        ("qkv_gate",256,(2,256,4,0,7,1,0,1)), ("qkv_gate",256,(2,256,4,0,6,1,0,1)),
        ("pooler",128,(2,128,6,0,4,1,0,1)), ("pooler",128,(1,128,8,0,2,1,1,0)),
        ("pooler",256,(2,256,4,0,4,1,0,1))]
ext = sk_build({tuple(c) + (0,) * (8 - len(c)) for _,_,c in CFGS})
weights = R.generate_weights(R.WEIGHTS_SEED)
runner = SkRunner(ext, weights["qkv_gate"].device)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)
ea = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
# empty-graph overhead
def capture(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize(); return g
z = torch.zeros(1, device='cuda')
ge = capture(lambda: z.add_(1))
ts = []
for _ in range(30):
    flush.add_(1); ea.record(); ge.replay(); eb.record(); eb.synchronize(); ts.append(ea.elapsed_time(eb)*1000)
print("TINY_GRAPH_US", round(statistics.median(ts), 2), flush=True)
for layer, m, cfg in CFGS:
    W = weights[layer]; n = W.shape[0]
    dd = R.generate_input(m, layer, seed=42); x = dd["x"]
    ok, msg = R.check_implementation(dd, runner.run(x, W, cfg), weights); print("CHECK", layer, m, cfg, ok, msg[:80], flush=True)
    ntile = n // (64 * cfg[0]); nct = ntile * cfg[4]
    runner.dbg = torch.zeros(nct * 6, dtype=torch.int64, device='cuda')
    g = capture(lambda: runner.run(x, W, cfg))
    evs = []
    for rep in range(10):
        flush.add_(1); torch.cuda.synchronize()
        ea.record(); g.replay(); eb.record(); eb.synchronize()
        evs.append(ea.elapsed_time(eb)*1000)
    d = runner.dbg.view(nct, 6).cpu().tolist()
    t0 = min(r[0] for r in d)
    rel = [[(v - t0)/1000 for v in r[:5]] + [r[5]] for r in d]
    def st(i): 
        v = [r[i] for r in rel]; return (round(min(v),2), round(statistics.median(v),2), round(max(v),2))
    print("TL", layer, m, cfg, "event_us", round(statistics.median(evs),2),
          "start", st(0), "first_tile", st(1), "mainloop_end", st(2), "sync_done", st(3), "end", st(4),
          "main_dur_med", round(statistics.median([r[2]-r[1] for r in rel]),2), flush=True)
    runner.dbg = torch.empty(0, dtype=torch.int64, device='cuda')
