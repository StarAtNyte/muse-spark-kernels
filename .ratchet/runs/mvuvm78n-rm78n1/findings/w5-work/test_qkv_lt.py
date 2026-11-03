"""Inspect cuBLASLt heuristic algorithms for T=16384 qkv on H200."""
import importlib.util, json, statistics, sys
import torch

sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)

lt = st["lt"]
ws = st["lt_ws"]
T = 16384
w_qkv = st["qkv_gate"]
N, K = w_qkv.shape
x = (torch.randn((T, K), device="cuda") * 0.5).to(torch.bfloat16)
out = torch.empty((T, N), dtype=torch.bfloat16, device="cuda")
ref = torch.mm(x, w_qkv.t()).float()

flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')

def cap(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    return g

def t1(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000

g_ref = cap(lambda: torch.mm(x, w_qkv.t()))
t_ref_samples = [t1(g_ref) for _ in range(10)]
t_ref = statistics.median(t_ref_samples)
print(f"torch.mm T={T} qkv: {t_ref:.2f} us", flush=True)

pid = lt.lt_plan(T, N, K, False, ws.numel(), 64)
n = lt.lt_nalgo(pid)
print(f"Number of heuristic algos returned: {n}", flush=True)

for a in range(n):
    try:
        info = list(lt.lt_info(pid, a))
        lt.lt_run(pid, a, x, w_qkv, out, ws)
        torch.cuda.synchronize()
        err = float((out.float() - ref).abs().max() / ref.abs().max())
        g_cand = cap(lambda a=a: lt.lt_run(pid, a, x, w_qkv, out, ws))
        ts = [t1(g_cand) for _ in range(10)]
        med = statistics.median(ts)
        print(f"Algo {a}: info={info} time={med:.2f} us (gain vs torch.mm = {t_ref / med:.4f}x, rel_err={err:.4e})", flush=True)
    except Exception as e:
        print(f"Algo {a} failed: {e}", flush=True)

print("DONE", flush=True)
