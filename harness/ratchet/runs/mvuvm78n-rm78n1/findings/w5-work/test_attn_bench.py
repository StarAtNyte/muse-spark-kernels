"""Compare our own attention kernel vs FA3 on the 8 scored shapes on H200."""
import importlib.util, json, statistics, sys
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
props = torch.cuda.get_device_properties(0)
nsm = props.multi_processor_count

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

# 8 scored cases
cases = [
    dict(load="turn1_slide", batch=1, ctx=80000, layer='sliding', q_len=753),
    dict(load="turn1_full",  batch=1, ctx=80000, layer='full',    q_len=753),
    dict(load="turn4_slide", batch=4, ctx=80000, layer='sliding', q_len=753),
    dict(load="turn4_full",  batch=4, ctx=80000, layer='full',    q_len=753),
    dict(load="ft_full_0",   batch=1, ctx=0,     layer='full',    q_len=16384),
    dict(load="ft_slide_32k",batch=1, ctx=32768, layer='sliding', q_len=16384),
    dict(load="ft_full_32k", batch=1, ctx=32768, layer='full',    q_len=16384),
    dict(load="ft_full_64k", batch=1, ctx=65536, layer='full',    q_len=16384),
]

for i, c in enumerate(cases):
    cc = dict(page=1, dist='soft', seed=9500 + i, **{k: v for k, v in c.items() if k != 'load'})
    if hasattr(R, 'case_fields'):
        cc.update(R.case_fields(cc, w))
    data = R.generate_input(**cc)
    cfg = data["config"]
    T = cfg["tokens"]
    q8 = torch.randn((T, 16, 64), dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    qkv = torch.randn((T, 2304), dtype=torch.bfloat16, device='cuda')

    # 1. Own kernel with tuned/pinned variant
    v_own = S._attn_variant(st, cfg)
    ns = S._attn_splits(cfg, data, v_own, nsm) if v_own is not None else 1
    def run_own():
        return S._attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v_own)

    # 2. FA3 path with fused norm_gate epilogue
    def run_fa3_ns0():
        return S._attention_norm_gate(data, q8, qkv, cfg)

    # 3. FA3 with num_splits = 1, 2, 4
    def make_fa3(nsplits):
        def _fn():
            B = cfg["batch"]
            window = cfg["window"]
            scaled = cfg["kv_scale"] != 1.0
            descale = {}
            if scaled:
                descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                               v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
            out = flash_attn_with_kvcache(
                q=q8.view(-1, 16, 64), k_cache=data["k_cache"], v_cache=data["v_cache"],
                page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
                cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=cfg["q_len"],
                softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                window_size=(window - 1, 0) if window else (-1, -1),
                scheduler_metadata=data["scheduler_metadata"], num_splits=nsplits, **descale)
            if isinstance(out, tuple): out = out[0]
            og = torch.empty((T, 1024), dtype=torch.bfloat16, device=q8.device)
            S._norm_gate_kernel[(T,)](out, qkv, og, num_warps=4)
            return og
        return _fn

    bench_fns = {
        f"own_v{v_own}_ns{ns}": run_own,
        "fa3_default": run_fa3_ns0,
        "fa3_ns1": make_fa3(1),
        "fa3_ns2": make_fa3(2),
        "fa3_ns4": make_fa3(4),
    }

    graphs = {}
    for name, fn in bench_fns.items():
        try:
            graphs[name] = cap(fn)
        except Exception as e:
            print(f"FAILED {c['load']} {name}: {e}", flush=True)

    timings = {k: [] for k in graphs}
    for rep in range(8):
        for k in list(graphs.keys()) if rep % 2 == 0 else list(graphs.keys())[::-1]:
            timings[k].append(t1(graphs[k]))

    res = {k: round(statistics.median(v), 1) for k, v in timings.items()}
    print(f"CASE {c['load']}: {json.dumps(res)}", flush=True)

print("ALL_DONE", flush=True)
