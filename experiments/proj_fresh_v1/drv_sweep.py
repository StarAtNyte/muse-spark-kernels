"""proj_fresh_v1 route sweep: every family (chenwainuo, vjeux, nb, ag) on every scored cell, prepared unpruned.
Reference checks (normal + outliers) per (cell, family), then interleaved cold-L2 graph replays (all families of a cell
in one rotating order per rep). Prints ROW json per (cell, family) with median us and the ratio to the _TABLE pick."""
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
FAM = ('chenwainuo', 'vjeux', 'nb', 'ag')
REPS = int(os.environ.get('REPS', '50'))
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')


def run(fam, d):
    S._ctx_bind(d['x'])
    return S._K[fam]['kernel_fn'](d, st[fam])


def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize(); return g


def tm(g):
    flush.add_(1.0); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000


cells = [(m, l) for m in (1, 16, 32, 8, 128, 256) for l in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')]
for m, l in cells:
    pick = S._TABLE.get('%s:%d' % (l, m))
    ok_f = []
    for fam in FAM:
        if fam == 'ag' and not S._K['ag']['has'](st['ag'], (l, m)):
            continue
        bad, msg = 0, ''
        try:
            for seed, dist in ((11, 'normal'), (12, 'outliers'), (13, 'normal')):
                d = R.generate_input(m, l, seed, dist=dist)
                out = run(fam, d); torch.cuda.synchronize()
                r = R.check_implementation(d, out, w)
                if not r[0]:
                    bad += 1; msg = str(r)[:160]
        except Exception as exc:  # noqa: BLE001
            bad, msg = 99, repr(exc)[:200]
        if bad:
            print('BAD', json.dumps({'m': m, 'layer': l, 'fam': fam, 'fails': bad, 'msg': msg}), flush=True)
        else:
            ok_f.append(fam)
    d = R.generate_input(m, l, 777)
    graphs = {}
    for fam in ok_f:
        try:
            graphs[fam] = cap(lambda f=fam: run(f, d))
        except Exception as exc:  # noqa: BLE001
            print('CAPERR', m, l, fam, repr(exc)[:200], flush=True)
    names = list(graphs)
    ts = {f: [] for f in names}
    for r in range(REPS):
        k = r % max(1, len(names))
        for f in names[k:] + names[:k]:
            ts[f].append(tm(graphs[f]))
    med = {f: statistics.median(v) for f, v in ts.items()}
    base = med.get(pick)
    for f in names:
        print('ROW', json.dumps({'m': m, 'layer': l, 'fam': f, 'pick': pick, 'us': round(med[f], 3),
                                 'ratio_vs_pick': round(med[f] / base, 4) if base else None}), flush=True)
    del graphs
    torch.cuda.empty_cache()
print('DONE %.0fs' % (time.time() - t0), flush=True)
