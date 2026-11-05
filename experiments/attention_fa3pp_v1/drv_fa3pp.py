"""(copy of experiments/attention_xu_offload_v1/drv_variants.py: variants from /w/modal/_fa3pp, build logs on,
flat q_len 8 kernel resource usage printed)
Paired screen of attention variants in /w/modal/_fa3pp/*.py (base first) on full-verify and
other scored shapes; every output reference-checked; cold L2 (read flush), CUDA graphs, rotating order."""
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
import torch.utils.cpp_extension as _ce
_li = _ce.load_inline
def _load_inline(*a, **k):
    k['extra_cuda_cflags'] = list(k.get('extra_cuda_cflags', [])) + ['-Xptxas=-warn-spills,-warn-lmem-usage']
    k['verbose'] = True
    m = _li(*a, **k)
    import glob, subprocess
    for so in glob.glob(f"{_ce._get_build_directory(k['name'], False)}/*.so"):
        r = subprocess.run(['cuobjdump', '--dump-resource-usage', so], capture_output=True, text=True)
        lines = r.stdout.split('\n')
        for i, l in enumerate(lines):
            if 'attn_kernelILi8ELb1' in l and i + 1 < len(lines):
                print('RES', k['name'], lines[i + 1].strip(), flush=True)
    return m
_ce.load_inline = _load_inline
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
NAMES = os.environ.get('VARIANTS', 'base,ctl,m1s8,pp1s8,np2s4,np1s8').split(',')
CASES = os.environ.get('CASES', 'v16f,v32f').split(',')
SEEDS = [int(s) for s in os.environ.get('SEEDS', '11,12').split(',')]
REPS = int(os.environ.get('REPS', '16'))
MODS = {n: load('m_' + n, f'/w/modal/_fa3pp/{n}.py') for n in NAMES}
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(M, d):
    o = M.kernel_fn(d); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = M.kernel_fn(d)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
SPEC = {}
for q, qn in ((1, 'd'), (8, 'v')):
    for bsz in (1, 16, 32):
        for layer, dist, ln in (('sliding', 'soft', 's'), ('full', 'coherent', 'f')):
            SPEC[f'{qn}{bsz}{ln}'] = dict(batch=bsz, ctx=80000, heads='tp8', layer=layer, page=1, dist=dist, q_len=q,
                                          spread=('agentic' if bsz > 1 else 'fixed'))
TESTS = dict(
    t5s=dict(batch=5, ctx=5000, heads='tp8', layer='sliding', q_len=8, spread='mixed', page=1),
    t2f=dict(batch=2, ctx=100000, heads='tp8', layer='full', q_len=8, page=1),
    t4e=dict(batch=4, ctx=6000, heads='tp8', layer='full', q_len=8, dist='edges'),
    t1m=dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
    t3k=dict(batch=3, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=0.015625),
    t4s=dict(batch=4, ctx=6000, heads='tp8', layer='sliding', q_len=8, dist='soft', page=1),
    t2ds=dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='soft'),
    t2dc=dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='coherent'))
for tn in [t for t in os.environ.get('TESTS', 't2f,t4e,t2ds,t2dc').split(',') if t]:
    for seed in SEEDS:
        d = R.generate_input(**TESTS[tn], seed=seed)
        r = {}
        for n, M in MODS.items():
            try:
                o = M.kernel_fn(d); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o)); r[n] = (ok, str(msg)[:70])
            except Exception as e:
                r[n] = (False, 'run failed: ' + str(e)[:150])
        print('TEST', json.dumps(dict(case=tn, seed=seed, res=r)), flush=True)
for cn in [c for c in CASES if c]:
    for seed in SEEDS:
        d = R.generate_input(**SPEC[cn], seed=seed)
        res = {}
        for n, M in MODS.items():
            try:
                g, o = cap(M, d); ok, msg = R.check_implementation(d, R.collect(d, o)); res[n] = (g, ok, str(msg)[:80])
            except Exception as e:
                res[n] = (None, False, 'run failed: ' + str(e)[:200])
                print('ERR', n, cn, str(e)[:400], flush=True)
        live = [n for n in NAMES if res[n][0] is not None]
        ts = {n: [] for n in live}
        for rep in range(REPS):
            order = live[rep % len(live):] + live[:rep % len(live)]
            if rep % 2: order = order[::-1]
            for n in order: ts[n].append(t1(res[n][0]))
        med = {n: statistics.median(v) for n, v in ts.items()}
        b = med.get('base')
        print('ROW', json.dumps(dict(case=cn, seed=seed, med={n: round(v, 2) for n, v in med.items()},
              gain={n: round(b / v, 3) for n, v in med.items()} if b else None,
              ok={n: res[n][1] for n in NAMES}, msg={n: res[n][2] for n in NAMES if not res[n][1]})), flush=True)
print('DONE', flush=True)
