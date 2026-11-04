"""claude_decoder_next_v1: MoE B4FIN (16 < T <= 32) final-pass readiness poll with __nanosleep backoff (-DOKC_MFS=48).
Parent (control) = scored_decoder_fp32_ybuf_v4, sha256 616187c9...; candidate = same file, only the b4 MoE build gets
-DOKC_MFS=48. One prepare, graphs captured per variant by swapping state['ma_ext']['b4']; reference-checked; ABBA
interleaved cold-L2 graph replays. Output bit-compared between variants."""
import importlib.util, json, statistics, sys, threading, time
from unittest.mock import MagicMock
import torch
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
sp = importlib.util.spec_from_file_location('cand', '/w/run/sub.py'); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
MFS = 48
box = {}
def build_cand():
    try:
        defs = tuple(S.MOE_DEFINES) + ('-DOKC_B4FIN=1', f'-DOKC_MFS={MFS}')
        e = S._RouterExt(S._ma_ext(defs + tuple(S._MA_SFX)))
        if S.MOE_HAS_PDL: e.okc_set_pdl(0)
        box['cand'] = e; log('candidate build done')
    except Exception as ex:
        box['err'] = repr(ex)[:300]; log('candidate build FAILED', box['err'])
th = threading.Thread(target=build_cand)
orig_prep = S._ma_prepare
def hook(weights, st):   # _MA_SFX is final here; build the candidate concurrently with the rest of prepare
    th.start(); return orig_prep(weights, st)
S._ma_prepare = hook
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
th.join()
log('PREPARED')
assert 'cand' in box, box
ctrl = S._moe(32, st)             # populates st['ma_ext']['b4'] with the parent build
assert st['ma_ext'].get('b4') is ctrl
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d, ext):
    st['ma_ext']['b4'] = ext
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
CASES = [dict(batch=32, ctx=80000, layer='sliding', page=1, dist='soft', spread='agentic', seed=8001),
         dict(batch=32, ctx=80000, layer='sliding', page=1, dist='soft', spread='agentic', seed=8002),
         dict(batch=32, ctx=80000, layer='full', page=1, dist='soft', spread='agentic', seed=8003),
         dict(batch=4, ctx=6000, layer='sliding', q_len=8, dist='soft', seed=8004),
         dict(batch=17, ctx=20000, layer='full', spread='mixed', page=1, seed=8005),
         dict(batch=24, ctx=30000, layer='sliding', spread='agentic', page=1, seed=8006)]
allg = []
for i, c in enumerate(CASES):
    if time.time() - T0 > 235: log('time guard: stopping before case', i); break
    ls = R.case_fields(c, w)['local_start']; d = R.generate_input(**c, local_start=ls)
    gc, oc = cap(d, ctrl); gk, ok_ = cap(d, box['cand'])
    chk = {}
    for nm, o in (('ctrl', oc), ('cand', ok_)):
        okk, msg = R.check_implementation(d, R.collect(d, o), w); chk[nm] = (bool(okk), str(msg)[:110])
    same = all(torch.equal(a, b) for a, b in zip(oc, ok_))
    tc, tk = [], []
    for rep in range(20):
        if rep % 2 == 0: tc.append(t1(gc)); tk.append(t1(gk))
        else: tk.append(t1(gk)); tc.append(t1(gc))
    mc, mk = statistics.median(tc), statistics.median(tk)
    print('ROW', json.dumps(dict(i=i, case=c, T=c['batch'] * c.get('q_len', 1), ctrl_us=round(mc, 2), cand_us=round(mk, 2),
          ratio_ctrl_over_cand=round(mc / mk, 4), min_ctrl=round(min(tc), 2), min_cand=round(min(tk), 2),
          ok=chk, bitwise_same=same)), flush=True)
    allg.append(mc / mk)
    del gc, gk, d, oc, ok_; torch.cuda.empty_cache()
if allg:
    log('GEOMEAN ratio ctrl/cand', round(statistics.geometric_mean(allg), 4), 'n', len(allg))
log('DONE')
