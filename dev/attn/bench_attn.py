import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_attn import *

def bench(batch, ctx, layer, q_len, splits, iters=10):
    data = R.generate_input(batch=batch, ctx=ctx, heads="tp8", layer=layer, seed=5, q_len=q_len, page=1, spread="fixed", dist="soft" if q_len == 1 else "normal")
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], 64
    tokens = batch * q_len
    q = torch.randn((tokens, hq, d), device=dev).to(torch.float8_e4m3fn)
    window = cfg["window"]
    span = window + q_len - 1 if window else data["page_table"].shape[1] * cfg["page_size"]
    chunk = triton.cdiv(triton.cdiv(span, splits), 16) * 16
    bm = max(16, triton.next_power_of_2(q_len * 8)); mt = bm // 16
    partial = torch.empty((batch, hkv, splits, bm, d), device=dev, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, bm), device=dev, dtype=torch.float32)
    def f():
        lib.fd_run(ctypes.c_int(mt), ctypes.c_int(4), ctypes.c_int(0), P(q), P(data["k_cache"]), P(data["v_cache"]), P(data["page_table"]), P(data["seq_lens"]),
                   P(data["k_scale"]), P(data["v_scale"]), P(partial), P(lse), ctypes.c_int(batch), ctypes.c_int(q_len), ctypes.c_int(hq),
                   ctypes.c_int(hkv), ctypes.c_int(data["page_table"].shape[1]), ctypes.c_int(1), ctypes.c_int(window), ctypes.c_int(splits),
                   ctypes.c_int(chunk), ctypes.c_float(0.1), ctypes.c_int(0), stream())
    f(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) / iters * 1000
    length = (window if window else ctx) 
    nbytes = batch * length * hkv * 64 * 2
    print(f"B={batch} ctx={ctx} {layer} L={q_len} splits={splits}: {us:.0f}us  {nbytes / us / 1e3:.0f} GB/s", flush=True)

for splits in (16, 32, 64):
    bench(8, 20000, "full", 1, splits)
bench(8, 20000, "full", 8, 32)
bench(4, 20000, "sliding", 1, 4)
