"""Paired full-short prefill attention timing on the same H200 inputs."""
import importlib.util, json, statistics, torch

spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
mod = sub._build(True)
st = {'mod': mod, 'attn_smem': sub.ATTN_SMEM,
      'no_pf': torch.empty((0,), dtype=torch.uint8, device='cuda')}
mod.attn_init(st['attn_smem'])
flush = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.int32, device='cuda')

def graph(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay(); torch.cuda.synchronize()
    return g, out

def elapsed(g):
    flush.add_(1)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000

nsm = torch.cuda.get_device_properties('cuda').multi_processor_count
for batch in (1, 4):
  for seed in (539, 1427):
    data, cfg, q8, qkv = sub._synthetic_attn_case(batch, 753, 80000, 0, 1, seed)
    graphs, outs, plans = {}, {}, {}
    for v in (43, 0, 1, 8, 9, 16, 24):
        n = sub._attn_splits(cfg, data, v, nsm)
        plans[v] = n
        graphs[v], outs[v] = graph(lambda v=v,n=n: sub._attn_custom(st, data, q8, qkv, cfg, nsplit=n, variant=v))
    for v in (0, 1, 8, 9, 16, 24):
        error = (outs[v].float() - outs[43].float()).norm() / outs[43].float().norm().clamp_min(1e-12)
        if not torch.isfinite(outs[v]).all().item() or error.item() > 0.03:
            print('FAIL', json.dumps({'batch':batch,'seed':seed,'variant':v,'relative_l2':error.item()}), flush=True)
            continue
        ratios, a, b = [], [], []
        for i in range(16):
            if i & 1: y, x = elapsed(graphs[v]), elapsed(graphs[43])
            else: x, y = elapsed(graphs[43]), elapsed(graphs[v])
            a.append(x); b.append(y); ratios.append(x / y)
        print('RESULT', json.dumps({'batch':batch,'seed':seed,'variant':v,'base_splits':plans[43], 'splits':plans[v], 'relative_l2':error.item(),
              'parent_us':statistics.median(a),'candidate_us':statistics.median(b),
              'speedup':statistics.median(ratios)}), flush=True)
