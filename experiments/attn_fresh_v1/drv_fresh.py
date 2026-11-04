"""Paired screen of /w/modal/_fresh/<name>.py variants (base first). PHASE=sass: build, resource usage + SASS opcode
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
        if 'attn_kernelILi8ELb1' in l and i + 1 < len(r):
            print('RES', name, l.strip()[-60:], r[i + 1].strip(), flush=True)
    txt = subprocess.run(['cuobjdump', '-sass', so], capture_output=True, text=True).stdout
    for fn in txt.split('Function : ')[1:]:
        head = fn.split('\n', 1)[0].strip()
        if 'attn_kernelILi8ELb1' not in head: continue
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
NAMES = os.environ.get('VARIANTS', 'base,g16,l32').split(',')
MODS = {}
for n in list(NAMES):
    try:
        M = load('m_' + n, f'/w/modal/_fresh/{n}.py'); M._module(); MODS[n] = M
    except Exception as e:
        print('BUILDFAIL', n, str(e)[-1500:], flush=True); NAMES.remove(n)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
OVR = json.loads(os.environ.get('OVR', '{}'))
def cap(M, d, ov=None):
    fn = (lambda: M._launch(d, **ov)) if ov else (lambda: M.kernel_fn(d))
    o = fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = fn()
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
if PHASE == 'stamp':
    M = MODS[os.environ.get('STAMPV', 'stamp')]
    for cn in [c for c in os.environ.get('CASES', 'd16f').split(',') if c]:
        d = R.generate_input(**SPEC[cn], seed=SEEDS[0])
        o = M.kernel_fn(d); torch.cuda.synchronize(); ok, msg = R.check_implementation(d, R.collect(d, o))
        print('STAMPCHK', cn, ok, str(msg)[:80], flush=True)
        for rep in range(REPS):
            dbg = torch.zeros(4096 * 16, dtype=torch.int64, device='cuda')
            flush.sum(); torch.cuda.synchronize()
            M._launch(d, dbg=dbg, **json.loads(os.environ.get('STAMPOV', '{}'))); torch.cuda.synchronize()
            G = int((dbg.view(-1, 16)[:, 0] > 0).sum())
            t = dbg.view(-1, 16)[:G].cpu().double()
            t0 = t[:, 0].min()
            st = ((t[:, 0] - t0) / 1000).tolist(); en = ((t[:, 1] - t0) / 1000).tolist(); fn_ = ((t[:, 2] - t0) / 1000).tolist()
            nseg = t[:, 3].tolist(); sm = t[:, 15].tolist()
            def q(v, f): v = sorted(v); return round(v[min(len(v) - 1, int(f * len(v)))], 2)
            per = collections.defaultdict(list)
            for i in range(G): per[int(sm[i])].append(en[i])
            smend = [max(v) for v in per.values()]
            e1 = [en[i] for i in range(G) if nseg[i] == 1]; e2 = [en[i] for i in range(G) if nseg[i] >= 2]
            seg1 = ((t[:, 4] - t0) / 1000).tolist(); mg = ((t[:, 14] - t0) / 1000).tolist()
            dur = [en[i] - mg[i] for i in range(G)]
            print('STAMP', json.dumps(dict(case=cn, rep=rep, G=G, start=[q(st, 0), q(st, .5), q(st, .9), q(st, 1)],
                  end=[q(en, 0), q(en, .1), q(en, .5), q(en, .9), q(en, 1)], fin=[q(fn_, .5), q(fn_, 1)],
                  smend=[q(smend, 0), q(smend, .1), q(smend, .5), q(smend, 1)],
                  end_nseg1=[q(e1, .5), q(e1, 1)] if e1 else None, end_nseg2=[q(e2, .5), q(e2, 1)] if e2 else None,
                  n2=len(e2), merge_start=[q(mg, .1), q(mg, .5), q(mg, 1)], last_merge_dur=[q(dur, .5), q(dur, 1)], seg1_end2=[q([seg1[i] for i in range(G) if nseg[i] >= 2], .5)] if e2 else None)), flush=True)
            if rep == 0 and os.environ.get('DUMPCTA'):
                print('CTA', cn, json.dumps([[int(sm[i]), round(st[i], 2), round(en[i], 2), int(nseg[i])] for i in range(G)]), flush=True)
elif PHASE == 'sass':
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
            for on, spec in OVR.items():
                if cn not in spec.get('cases', [cn]): continue
                M = MODS[spec.get('of', 'base')]
                try:
                    g, o = cap(M, d, spec['ov']); ok, msg = R.check_implementation(d, R.collect(d, o)); res[on] = (g, ok, str(msg)[:80])
                except Exception as e:
                    res[on] = (None, False, 'run failed: ' + str(e)[:200])
            live = [n for n in res if res[n][0] is not None and (res[n][1] or os.environ.get('ALLOWBAD'))]
            ts = {n: [] for n in live}
            for rep in range(REPS):
                order = live[rep % len(live):] + live[:rep % len(live)]
                if rep % 2: order = order[::-1]
                for n in order: ts[n].append(t1(res[n][0]))
            med = {n: statistics.median(v) for n, v in ts.items()}
            b = med.get('base')
            print('ROW', json.dumps(dict(case=cn, seed=seed, med={n: round(v, 2) for n, v in med.items()},
                  gain={n: round(b / v, 3) for n, v in med.items()} if b else None,
                  bad={n: res[n][2] for n in res if not res[n][1]})), flush=True)
print('DONE', flush=True)
