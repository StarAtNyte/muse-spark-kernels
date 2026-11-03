
# ---------------- driver ----------------
import importlib.util, json, os, statistics, sys, time
import torch.nn.functional as F
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R

CONFIGS = {
    ("qkv_gate", 128): [(2,128,6,0,4,1,1), (2,128,6,0,6,1,1), (2,128,6,0,7,1,1), (3,128,5,0,8,1,1), (1,128,8,0,3,1,1), (1,128,8,0,4,1,1)],
    ("qkv_gate", 256): [(2,256,4,0,4,1,1), (2,256,4,0,6,1,1), (2,256,4,0,7,1,1), (1,256,5,0,3,1,1), (1,256,5,0,4,1,1)],
    ("pooler", 128):   [(2,128,6,0,4,1,1), (2,128,6,0,2,1,1), (1,128,8,0,2,1,1), (2,128,6,0,8,1,1)],
    ("pooler", 256):   [(2,256,4,0,4,1,1), (2,256,4,0,2,1,1), (1,256,5,0,2,1,1), (2,256,4,0,8,1,1)],
}
ONLY = os.environ.get("ONLY")
if ONLY:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k[0] in ONLY.split(",")}
t0 = time.time()
ext = sk_build({c[:7] for v in CONFIGS.values() for c in v})
print(f"built in {time.time()-t0:.1f}s", flush=True)
weights = R.generate_weights(R.WEIGHTS_SEED)
dev = weights["qkv_gate"].device
runner = SkRunner(ext, dev)

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)
ea = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
def capture(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); torch.cuda.synchronize()
    return g, out
def elapsed(g):
    flush.add_(1); ea.record(); g.replay(); eb.record(); eb.synchronize()
    return ea.elapsed_time(eb) * 1000.0
def paired(graphs, reps=40):
    t = [[] for _ in graphs]
    for r in range(reps):
        order = range(len(graphs)) if r % 2 == 0 else reversed(range(len(graphs)))
        for i in order:
            t[i].append(elapsed(graphs[i]))
    return [statistics.median(v) for v in t]

def screen(base_fn, label):
    rows = []
    for (layer, m), cfgs in CONFIGS.items():
        n = weights[layer].shape[0]
        sol = R.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989.5e12}) / 1000.0
        good = []
        for cfg in cfgs:
            try:
                for seed, dist in ((123, "normal"), (7, "outliers")):
                    d = R.generate_input(m, layer, seed=seed, dist=dist)
                    o = runner.run(d["x"], weights[layer], cfg)
                    torch.cuda.synchronize()
                    ok, msg = R.check_implementation(d, o, weights)
                    if not ok:
                        raise RuntimeError(msg)
                good.append(cfg)
            except Exception as e:
                print(f"CFG_FAIL {layer} m={m} {cfg}: {e}", flush=True)
        per = {}
        for seed in (42, 108, 999):
            d = R.generate_input(m, layer, seed=seed)
            gs = [capture(lambda: base_fn(d))[0]]
            for cfg in good:
                gs.append(capture(lambda cfg=cfg: runner.run(d["x"], weights[layer], cfg))[0])
            meds = paired(gs)
            for i, t in enumerate(meds):
                per.setdefault(i, []).append(t)
            del gs
        bt = statistics.geometric_mean(per[0])
        row = {"layer": layer, "m": m, "sol": round(sol, 2), label: round(bt, 2),
               "cands": {str(c): round(statistics.geometric_mean(per[i + 1]), 2) for i, c in enumerate(good)}}
        print("SCREEN_ROW", json.dumps(row), flush=True)
        rows.append(row)
    return rows

def cublas(d):
    return F.linear(d["x"], weights[d["config"]["layer"]])
screen(cublas, "cublas")

if os.environ.get("BASE", "1") == "1":
    spec = importlib.util.spec_from_file_location("cand", "/w/run/sub.py")
    cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
    t0 = time.time(); state = cand.prepare(weights); print(f"base prepare {time.time()-t0:.1f}s", flush=True)
    screen(lambda d: cand.kernel_fn(d, state), "base")
