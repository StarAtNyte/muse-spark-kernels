"""ncu target: one prefill layer call (first turn, 16384 tokens, full) so the fused MoE kernel can be profiled.
Attention choice pinned after prepare (the in-prepare FA3 validation cannot run on Modal)."""
import importlib.util, sys
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
sys.path.insert(0, '/w/kernels/muse_prefill_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
old = """          int v;
          while (true) {
            asm volatile(\"ld.acquire.gpu.global.b32 %0, [%1];\" : \"=r\"(v) : \"l\"(fa.done + tile) : \"memory\");
            if (v >= need) break;
            __nanosleep(64);
          }
          asm volatile(\"fence.proxy.async.global;\" ::: \"memory\");"""
new = """          unsigned long long wait0 = 0, wait1 = 0;
          if (fa.dbg) asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(wait0));
          int v;
          while (true) {
            asm volatile(\"ld.acquire.gpu.global.b32 %0, [%1];\" : \"=r\"(v) : \"l\"(fa.done + tile) : \"memory\");
            if (v >= need) break;
            __nanosleep(64);
          }
          if (fa.dbg) {
            asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(wait1));
            fa.dbg[(size_t)nitems * 4 + item] = wait1 - wait0;
          }
          asm volatile(\"fence.proxy.async.global;\" ::: \"memory\");"""
assert S._CUDA_SRC.count(old) == 1
S._CUDA_SRC = S._CUDA_SRC.replace(old, new)
import json, statistics, zlib, base64
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
st['attn_choice'] = dict(S.ATTN_FIXED)
for seed in (9400, 9407, 9417):
    c = dict(batch=1, ctx=80000, layer='sliding', page=1, dist='soft', q_len=753, seed=seed)
    if hasattr(R, 'case_fields'): c.update(R.case_fields(c, w))
    d = R.generate_input(**c)
    maxitems = S._max_tiles(c['batch'] * c['q_len']) * 96
    st['moe_dbg'] = torch.zeros(5 * maxitems, device='cuda', dtype=torch.int64)
    out = S.kernel_fn(d, st); torch.cuda.synchronize()
    ok,msg=R.check_implementation(d,R.collect(d,out),w)
    print('CHECK',seed,ok,str(msg)[:200],flush=True)
    if not ok: raise AssertionError(msg)
    buf=st['moe_dbg'].cpu().tolist(); events=[]
    for i in range(maxitems):
        a,b,packed,cta=buf[4*i:4*i+4]
        if not a: break
        if b<a: raise ValueError(('bad trace',i,a,b))
        events.append((a,b,packed,cta))
    n=len(events)
    assert n and n % 96 == 0 and (n == maxitems or buf[4*n] == 0), (n,maxitems)
    first=n*2//3
    waits=buf[4*n+first:5*n]
    assert len(waits)==n-first
    positive=sorted(waits)
    summary={'seed':seed,'ntiles':n//96,'items':n,'phase2_items':len(waits),
             'wait_median_ns':statistics.median(waits),'wait_p90_ns':positive[int(.9*(len(waits)-1))],
             'wait_max_ns':max(waits),'wait_over_1us':sum(x>1000 for x in waits),
             'span_us':(max(e[1] for e in events)-min(e[0] for e in events))/1000}
    raw=json.dumps({'seed':seed,'events':events,'phase2_wait_ns':waits},separators=(',',':')).encode()
    print('SUMMARY',json.dumps(summary),flush=True)
    print('TRACE_ZLIB_BASE64',base64.b64encode(zlib.compress(raw,9)).decode(),flush=True)
print('DONE',flush=True)
