"""proj_fresh_v1 wave screen in the real pruned file: parent dispatcher (kernel_fn) vs persistent cgemvp at several
CTAs-per-SM, per m=1 cgemv cell. Reference checks per config; interleaved cold-L2 graph replays (rotating order)."""
import importlib.util, json, os, statistics, sys, time
from unittest.mock import MagicMock
import torch


class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m

    def exec_module(self, m):
        pass


sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_bf16_proj')
import reference as R
t0 = time.time()
sp = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
print('PREP done %.0fs' % (time.time() - t0), flush=True)
ext = S._K['vjeux']['warm']()
vst = st['vjeux']
nsm = torch.cuda.get_device_properties(0).multi_processor_count
CFG = {'qkv_gate': (2, 256, 2304), 'o': (2, 64, 8192), 'unpooler': (2, 128, 1024), 'pooler': (2, 512, 4096)}
GRIDS = {'qkv_gate': [2, 3, 4, 5, 6, 7, 0], 'o': [4, 6, 8, 12, 16, 0], 'unpooler': [1, 2, 3, 0], 'pooler': [1, 2, 3, 0]}
REPS = int(os.environ.get('REPS', '60'))
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize(); return g


def tm(g):
    flush.add_(1.0); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000


def cgp(x, layer, k):
    rows, nth, n = CFG[layer]
    cg = vst['cg'][layer]
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    grid = k * nsm if k else n // rows
    ext.cgemvp(cg['L'], cg['C'], x, cg['tab'], cg['pmax'], out, cg['base4'], rows, nth, grid)
    return out


print('CHOICE', {l: str(vst.get('choice', {}).get((l, 1))) for l in CFG}, flush=True)
for layer in CFG:
    fns = {'parent': lambda d: S.kernel_fn(d, st)}
    for k in GRIDS[layer]:
        fns['p%d' % k] = (lambda d, kk=k, ll=layer: cgp(d['x'], ll, kk))
    good = []
    for name, fn in fns.items():
        bad, msg = 0, ''
        try:
            for seed, dist in ((11, 'normal'), (12, 'outliers'), (13, 'normal')):
                d = R.generate_input(1, layer, seed, dist=dist)
                out = fn(d); torch.cuda.synchronize()
                r = R.check_implementation(d, out, w)
                if not r[0]:
                    bad += 1; msg = str(r)[:160]
                ref = S.kernel_fn(d, st); torch.cuda.synchronize()
                if name != 'parent' and not torch.equal(out, ref):
                    msg += ' (not bit-identical to parent)'
        except Exception as exc:  # noqa: BLE001
            bad, msg = 99, repr(exc)[:200]
        print('CHK', json.dumps({'layer': layer, 'cfg': name, 'fails': bad, 'msg': msg}), flush=True)
        if not bad:
            good.append(name)
    d = R.generate_input(1, layer, 777)
    gs = {nm: cap(lambda f=fns[nm]: f(d)) for nm in good}
    names = list(gs); ts = {nm: [] for nm in names}
    for r in range(REPS):
        k = r % len(names)
        for nm in names[k:] + names[:k]:
            ts[nm].append(tm(gs[nm]))
    med = {nm: statistics.median(v) for nm, v in ts.items()}
    for nm in names:
        print('ROW', json.dumps({'layer': layer, 'cfg': nm, 'us': round(med[nm], 3),
                                 'ratio': round(med[nm] / med['parent'], 4) if 'parent' in med else None}), flush=True)
    del gs
print('DONE %.0fs' % (time.time() - t0), flush=True)
