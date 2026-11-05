"""Paired screen of /w/modal/_f16acc/<name>.py variants (base first). PHASE=sass: build, resource usage + SASS opcode
histogram of the flat q_len-8 kernel (whole function and its HGMMA region), reference checks on the TASK.md tests.
PHASE=time: cold-L2 CUDA-graph replays, order rotated each rep, every graph reference-checked."""
import importlib.util, json, os, statistics, sys, re, collections, glob, subprocess
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
PHASE = os.environ.get('PHASE', 'time')
DUMP = os.environ.get('DUMP', '')
_li = _ce.load_inline
def sass_report(name, so):
    r = subprocess.run(['cuobjdump', '--dump-resource-usage', so], capture_output=True, text=True).stdout.split('\n')
    for i, l in enumerate(r):
        if os.environ.get('KPAT', 'attn_kernelILi8E') in l and i + 1 < len(r):
            print('RES', name, l.strip()[-60:], r[i + 1].strip(), flush=True)
    txt = subprocess.run(['cuobjdump', '-sass', so], capture_output=True, text=True).stdout
    for fn in txt.split('Function : ')[1:]:
        head = fn.split('\n', 1)[0].strip()
        if os.environ.get('KPAT', 'attn_kernelILi8E') not in head: continue
        ins = []
        for l in fn.split('\n'):
            m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*?);', l)
            if m: ins.append((m.group(3), (m.group(2) or '') + m.group(3) + m.group(4)))
        ops = [o for o, _ in ins]
        hg = [k for k, o in enumerate(ops) if 'GMMA' in o]
        reg = ins[hg[0]:hg[-1] + 1] if hg else []
        def hist(lst):
            c = collections.Counter(o.split('.')[0] for o, _ in lst); return dict(c.most_common())
        print('SASS', name, head[-50:], json.dumps(dict(total=len(ins), region=len(reg), hist_region=hist(reg))), flush=True)
        if DUMP and any(x in name for x in DUMP.split(',')):
            for o, full in reg: print('  ', re.sub(r'\s+', ' ', full)[:110])
def _load_inline(*a, **k):
    k['extra_cuda_cflags'] = list(k.get('extra_cuda_cflags', [])) + ['-Xptxas=-warn-spills,-warn-lmem-usage']
    m = _li(*a, **k)
    if PHASE == 'sass':
        for so in glob.glob(f"{_ce._get_build_directory(k['name'], False)}/*.so"): sass_report(k['name'], so)
    return m
_ce.load_inline = _load_inline
def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
NAMES = os.environ.get('VARIANTS', 'base,f16').split(',')
MODS = {}
for n in list(NAMES):
    try:
        M = load('m_' + n, f'/w/modal/_f16acc/{n}.py'); M._module(); MODS[n] = M
    except Exception as e:
        print('BUILDFAIL', n, str(e)[-1500:], flush=True); NAMES.remove(n)
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
TESTS = [
    dict(batch=1, ctx=10, heads='tp8', layer='sliding'),
    dict(batch=3, ctx=2047, heads='tp8', layer='sliding'),
    dict(batch=5, ctx=5000, heads='tp8', layer='sliding', q_len=8, spread='mixed', page=1),
    dict(batch=4, ctx=3000, heads='tp8', layer='full'),
    dict(batch=8, ctx=8192, heads='dp', layer='full', spread='mixed'),
    dict(batch=2, ctx=100000, heads='tp8', layer='full', q_len=8, page=1),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', dist='edges', page=1),
    dict(batch=4, ctx=6000, heads='tp8', layer='full', q_len=8, dist='edges'),
    dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
    dict(batch=3, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=9000, heads='tp8', layer='full', kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='soft'),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', q_len=8, dist='soft', page=1),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='soft'),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='coherent', page=1),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='coherent'),
]
SEEDS = [int(s) for s in os.environ.get('SEEDS', '11,12').split(',')]
REPS = int(os.environ.get('REPS', '20'))
if PHASE == 'sass':
    sel = [int(x) for x in os.environ.get('TESTS', ','.join(str(i) for i in range(16))).split(',') if x]
    extra = [SPEC[c] for c in os.environ.get('XCASES', 'v16f,v1f').split(',') if c]
    for ti, tc in [(i, TESTS[i]) for i in sel] + [('S' + str(j), c) for j, c in enumerate(extra)]:
        for seed in SEEDS:
            d = R.generate_input(**tc, seed=seed); r = {}
            for n, M in MODS.items():
                try:
                    o = M.kernel_fn(d); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o))
                    m = re.search(r'error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\) \[bound ([0-9.]+)', str(msg))
                    r[n] = (ok, m.groups() if m else str(msg)[:90])
                except Exception as e:
                    r[n] = (False, 'run failed: ' + str(e)[:150])
            print('TEST', ti, seed, json.dumps(r), flush=True)
else:
    for cn in [c for c in os.environ.get('CASES', 'v16f,v32f').split(',') if c]:
        for seed in SEEDS:
            d = R.generate_input(**SPEC[cn], seed=seed); res = {}
            for n, M in MODS.items():
                try:
                    g, o = cap(M, d); ok, msg = R.check_implementation(d, R.collect(d, o)); res[n] = (g, ok, str(msg)[:80])
                except Exception as e:
                    res[n] = (None, False, 'run failed: ' + str(e)[:200])
            live = [n for n in NAMES if res[n][0] is not None and (res[n][1] or os.environ.get('ALLOWBAD'))]
            ts = {n: [] for n in live}
            for rep in range(REPS):
                order = live[rep % len(live):] + live[:rep % len(live)]
                if rep % 2: order = order[::-1]
                for n in order: ts[n].append(t1(res[n][0]))
            med = {n: statistics.median(v) for n, v in ts.items()}
            b = med.get(os.environ.get('BASE', 'base'))
            print('ROW', json.dumps(dict(case=cn, seed=seed, med={n: round(v, 2) for n, v in med.items()},
                  gain={n: round(b / v, 3) for n, v in med.items()} if b else None,
                  bad={n: res[n][2] for n in NAMES if not res[n][1]})), flush=True)
print('DONE', flush=True)
