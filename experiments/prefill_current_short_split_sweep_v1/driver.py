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

for seed in (539, 1427, 7919):
    data, cfg, q8, qkv = sub._synthetic_attn_case(1, 753, 80000, 0, 1, seed)
    graphs, outs = {}, {}
    for n in (2, 4, 5, 6, 8):
        graphs[n], outs[n] = graph(lambda n=n: sub._attn_custom(st, data, q8, qkv, cfg, nsplit=n, variant=43))
    for n in (4, 5, 6, 8):
        error = (outs[n].float() - outs[2].float()).norm() / outs[2].float().norm()
        if not torch.isfinite(outs[n]).all().item() or error.item() > 0.03:
            raise AssertionError((seed, n, error.item()))
        ratios, a, b = [], [], []
        for i in range(16):
            if i & 1: y, x = elapsed(graphs[n]), elapsed(graphs[2])
            else: x, y = elapsed(graphs[2]), elapsed(graphs[n])
            a.append(x); b.append(y); ratios.append(x / y)
        print('RESULT', json.dumps({'seed':seed,'splits':n,'relative_l2':error.item(),
              'parent_us':statistics.median(a),'candidate_us':statistics.median(b),
              'speedup':statistics.median(ratios)}), flush=True)
