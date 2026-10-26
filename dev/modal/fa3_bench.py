import sys, os, torch
sys.path.insert(0, "/w/kernels/muse_attention_decode")
import reference as R
from sgl_kernel.flash_attn import flash_attn_with_kvcache
FP8 = torch.float8_e4m3fn
flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=12):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]
for B, QL in ((16, 8), (32, 8), (16, 1), (32, 1)):
    d = R.generate_input(B, 80000, "tp8", "full", 3, q_len=QL, spread="agentic", dist="coherent", page=1)
    cfg = d["config"]; hq, hkv, dd = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = B * QL
    q = torch.randn(tokens, hq, dd, device="cuda").to(FP8)
    def fa3():
        return flash_attn_with_kvcache(q=q, k_cache=d["k_cache"], v_cache=d["v_cache"], page_table=d["page_table"],
            cache_seqlens=d["cache_seqlens"], cu_seqlens_q=d["cu_seqlens_q"], max_seqlen_q=QL,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True, window_size=(-1, -1),
            scheduler_metadata=d["scheduler_metadata"])
    print("FA3 full B=%d QL=%d: %.1f us" % (B, QL, timeit(fa3)), flush=True)
