"""Paired cold-L2 graph screen of compile-time variants of /w/run/sub.py's MAIN (WGS) build vs the unmodified defines.
VARIANTS env: ';'-separated, each a ','-separated list of extra -D flags. BSS env: batch sizes. SEEDS env."""
import importlib.util, json, os, statistics, sys, time
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('START', flush=True)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
sfx = st.get('sfx', ())
sfx = tuple(sfx)
def defs_for(M):
    if M <= S.TINY_MAX_M: return tuple(S.TINY_DEFINES) + sfx
    if M <= S.TINY2_MAX_M: return tuple(S.TINY2_DEFINES) + sfx
    if M <= S.MID_MAX_M: return tuple(S.MID_DEFINES) + sfx
    return tuple(S._main_defines(sfx)) + sfx
variants = [tuple(v.split(',')) for v in os.environ.get('VARIANTS', '-DOKC_WPF=4').split(';') if v]
grid = st['grid']
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)

def capture(ext, data):
    x = data['hidden_states'].contiguous(); ids = data['topk_ids'].to(torch.int32).contiguous(); tw = data['topk_weights'].float().contiguous()
    out = torch.empty_like(x)
    def f(): ext.okc_run(st['w'], st['scratch'], x, ids, tw, out, 0, 32, grid)
    f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize()
    return g, out, x, ids, tw

def timed(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000

REPS = int(os.environ.get('REPS', '30'))
for bs in [int(b) for b in os.environ.get('BSS', '32,128,256').split(',')]:
    for seed in [int(s) for s in os.environ.get('SEEDS', '403').split(',')]:
        d = R.generate_input(bs, 'ep8', seed)
        with torch.no_grad(): ref = R.ref_kernel(d, w)
        t0 = time.time(); bd = defs_for(bs)
        exts = {(): S._ext(bd)}
        for v in variants: exts[v] = S._ext(bd + v)
        print('COMPILED', bs, round(time.time() - t0, 1), flush=True)
        graphs = {}
        for k, e in exts.items():
            graphs[k] = capture(e, d)
            ok, msg = compare(graphs[k][1], ref, R.REL_L2_TOL)
            if not ok:
                if os.environ.get('ALLOWBAD'): print('BAD', k, bs, msg, flush=True)
                else: raise AssertionError(f'{k} bs={bs}: {msg}')
        same = {str(k): bool(torch.equal(graphs[k][1], graphs[()][1])) for k in exts if k}
        base = graphs[()][0]
        for k in exts:
            if not k: continue
            tb = []; tc = []
            for rep in range(REPS):
                if rep & 1: tc.append(timed(graphs[k][0])); tb.append(timed(base))
                else: tb.append(timed(base)); tc.append(timed(graphs[k][0]))
            print('ROW', json.dumps({'bs': bs, 'seed': seed, 'var': ' '.join(k), 'base_us': round(statistics.median(tb), 2),
                  'cand_us': round(statistics.median(tc), 2), 'gain': round(statistics.median(tb) / statistics.median(tc), 4),
                  'bitwise': same[str(k)]}), flush=True)
        del graphs, ref, d
print('DONE', flush=True)
