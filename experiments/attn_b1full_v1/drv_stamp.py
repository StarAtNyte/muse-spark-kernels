"""Batch-1 attention latency breakdown: per-CTA globaltimer stamps (stamp variant) + event timings of
empty graph / 1-element kernel graph / base, plus bracket kernels (globaltimer before/after) in one graph."""
import importlib.util, json, os, statistics, sys
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_attention_decode')
import reference as R
from torch.utils.cpp_extension import load_inline
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
TS = load_inline(name='tsk', cpp_sources='void ts(torch::Tensor t, int64_t i);', cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void tsk(long long* p, int i) { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); p[i] = t; }
void ts(torch::Tensor t, int64_t i) { tsk<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((long long*)t.data_ptr<int64_t>(), (int)i); }
''', functions=['ts'], verbose=False)
B = load('m_base', '/w/modal/_b1full/base.py')
S = load('m_stamp', '/w/modal/_b1full/' + os.environ.get('STAMPV', 'stamp') + '.py')
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
REPS = int(os.environ.get('REPS', '30'))
SPEC = {
  'd1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=1),
  'd1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=1),
  'v1s': dict(batch=1, ctx=80000, heads='tp8', layer='sliding', page=1, dist='soft', q_len=8),
  'v1f': dict(batch=1, ctx=80000, heads='tp8', layer='full', page=1, dist='coherent', q_len=8),
}
def graph(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
one = torch.zeros(1, device='cuda')
g_empty = graph(lambda: None) if False else None
for cn in os.environ.get('CASES', 'd1s,v1s,d1f,v1f').split(','):
    d = R.generate_input(**SPEC[cn], seed=11)
    o = B.kernel_fn(d); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o))
    dbg = torch.zeros(1024 * 16, dtype=torch.int64, device='cuda')
    br = torch.zeros(4, dtype=torch.int64, device='cuda')
    gb = graph(lambda: B.kernel_fn(d))
    gs = graph(lambda: S._launch(d, dbg=dbg))
    gs0 = graph(lambda: S.kernel_fn(d))
    gt = graph(lambda: one.add_(1))
    gbr = graph(lambda: (TS.ts(br, 0), S._launch(d, dbg=dbg), TS.ts(br, 1)))
    gtt = graph(lambda: (TS.ts(br, 2), TS.ts(br, 3)))
    ts = {k: [] for k in ('base', 'stamp_nodbg', 'stamp_dbg', 'tiny', 'bracket')}
    stamps = []
    for rep in range(REPS):
        ts['base'].append(t1(gb)); ts['stamp_nodbg'].append(t1(gs0)); ts['tiny'].append(t1(gt))
        dbg.zero_(); ts['stamp_dbg'].append(t1(gs))
        a = dbg.view(-1, 16).cpu()
        live = a[a[:, 0] > 0]
        stamps.append(live)
        ts['bracket'].append(t1(gbr)); t1(gtt)
        b4 = br.cpu().tolist()
        stamps[-1] = (live, b4)
    med = {k: round(statistics.median(v), 2) for k, v in ts.items()}
    # per-stamp distribution relative to the earliest CTA start, median over reps
    rows = {}
    ncta = None
    for k in (0, 1, 2, 3, 9, 4, 5, 8, 6, 7):
        mins, meds, maxs = [], [], []
        for live, _ in stamps:
            t0 = live[:, 0].min().item()
            v = live[:, k]; v = v[v > 0]
            if len(v) == 0: continue
            v = (v - t0).double() / 1000
            mins.append(v.min().item()); meds.append(v.median().item()); maxs.append(v.max().item())
            ncta = len(live)
        if mins:
            rows[k] = (round(statistics.median(mins), 2), round(statistics.median(meds), 2), round(statistics.median(maxs), 2), len(mins))
    pre, post = [], []
    for live, b4 in stamps:
        pass
    # bracket: separate replays after the stamp replay; dbg holds the bracket run's stamps too (same buffer)
    print('ROW', json.dumps(dict(case=cn, ok=ok, msg=str(msg)[:60], ncta=ncta, med=med, stamps=rows)), flush=True)
    # bracket analysis on a fresh replay
    br_pre, br_post, br_tt, span = [], [], [], []
    for rep in range(REPS):
        dbg.zero_(); t1(gbr); a = dbg.view(-1, 16).cpu(); live = a[a[:, 0] > 0]; b4 = br.cpu().tolist()
        t1(gtt); b5 = br.cpu().tolist()
        br_pre.append((live[:, 0].min().item() - b4[0]) / 1000)
        br_post.append((b4[1] - live[:, 7].max().item()) / 1000)
        span.append((live[:, 7].max().item() - live[:, 0].min().item()) / 1000)
        br_tt.append((b5[3] - b5[2]) / 1000)
    print('BRK', json.dumps(dict(case=cn, pre=round(statistics.median(br_pre), 2), post=round(statistics.median(br_post), 2),
          span=round(statistics.median(span), 2), tiny_to_tiny=round(statistics.median(br_tt), 2))), flush=True)
    # smid / per-CTA dump of the last rep
    live, _ = stamps[-1]
    t0 = live[:, 0].min().item()
    print('CTAS', cn, json.dumps([[int(r[15])] + [round((r[k].item() - t0) / 1000, 2) if r[k] > 0 else None for k in (0, 1, 2, 3, 9, 4, 5, 8, 6, 7)] for r in live[:300]]), flush=True)
print('DONE', flush=True)
