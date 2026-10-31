"""Paired full-short prefill attention timing on the same H200 inputs."""
import importlib.util, json, statistics, torch

def load(name):
    spec = importlib.util.spec_from_file_location(name, '/w/run/sub.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sub = load('base')
candidate = load('interleaved')
old = """  const int qb = p.nqb - 1 - blockIdx.x;          // heaviest (latest) query blocks first
  const int h = blockIdx.y & 1, split = blockIdx.y >> 1, b = blockIdx.z;"""
new = """  const int linear = blockIdx.x + blockIdx.y * gridDim.x;
  const int qb = p.nqb - 1 - (p.nsplit == 1 ? linear / 2 : (int)blockIdx.x);
  const int h = p.nsplit == 1 ? (linear & 1) : (blockIdx.y & 1);
  const int split = blockIdx.y >> 1, b = blockIdx.z;"""
assert candidate._CUDA_SRC.count(old) == 1
candidate._CUDA_SRC = candidate._CUDA_SRC.replace(old, new)
mod = sub._build(True)
cmod = candidate._build(True)
st = {'mod': mod, 'attn_smem': sub.ATTN_SMEM, 'no_pf': torch.empty((0,), dtype=torch.uint8, device='cuda')}
cs = {'mod': cmod, 'attn_smem': candidate.ATTN_SMEM, 'no_pf': torch.empty((0,), dtype=torch.uint8, device='cuda')}
mod.attn_init(st['attn_smem']); cmod.attn_init(cs['attn_smem'])
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

for ctx in (32768, 65536):
  for seed in (539, 1427):
    data, cfg, q8, qkv = sub._synthetic_attn_case(1, 16384, ctx, 0, 1, seed)
    gb, ob = graph(lambda: sub._attn_custom(st, data, q8, qkv, cfg, nsplit=1, variant=43))
    gc, oc = graph(lambda: candidate._attn_custom(cs, data, q8, qkv, cfg, nsplit=1, variant=43))
    err = (ob.float() - oc.float()).norm() / ob.float().norm().clamp_min(1e-12)
    if not torch.isfinite(oc).all().item() or err.item() > 0.001:
        raise AssertionError(('output difference', ctx, seed, err.item()))
    ratios, bt, ct = [], [], []
    for i in range(12):
        if i & 1: c, b = elapsed(gc), elapsed(gb)
        else: b, c = elapsed(gb), elapsed(gc)
        bt.append(b); ct.append(c); ratios.append(b / c)
    print('RESULT', json.dumps({'ctx':ctx, 'seed':seed, 'relative_l2':err.item(),
          'parent_us':statistics.median(bt), 'candidate_us':statistics.median(ct),
          'speedup':statistics.median(ratios)}), flush=True)
