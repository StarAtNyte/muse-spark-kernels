"""Decoder: sk2 weight stages issued before griddepcontrol.wait (BFP_PDL_N 2 = scored) vs 4/8/16, whole layer,
reference-checked, paired cold-L2 graph replays (variants captured from one module by switching the global)."""
import importlib.util, json, statistics, sys
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
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
def cap(d):
    o = S.kernel_fn(d, st); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): o = S.kernel_fn(d, st)
    g.replay(); torch.cuda.synchronize(); return g, o
def t1(g):
    flush.add_(1); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize(); return a.elapsed_time(b) * 1000
original_pdl = S._pdl
orig_cluster = dict(S.SL_CLUSTER_CONFIG)
orig_tile = dict(S.SL_TILE_CONFIG)
def select(name):
    S.BFP_PDL_N = 1 if name == 'preload1' else 2
    S.GLE_RC = {1: 4 if name == 'carry4' else 8}
    S.GLE_ILV = 2 if name == 'interleave2' else 1
    S.SL_CLUSTER_CONFIG = dict(orig_cluster)
    S.SL_TILE_CONFIG = dict(orig_tile)
    S._pdl = (lambda T: original_pdl(T) and T <= 8) if name == 'serial16_32' else original_pdl
    if name == 'finish4':
        S.SL_CLUSTER_CONFIG = {1:(4,256),8:(4,1024)}
    if name == 'route2': S.SL_TILE_CONFIG[8] = (4,2,True)
for i,(bsz,q,layer) in enumerate(( (1,1,'sliding'),(1,1,'full'),(1,8,'sliding'),(1,8,'full'),(16,1,'sliding'),(16,1,'full'),(32,1,'sliding'),(32,1,'full') )):
    V = ['base','preload1'] + (['carry4','interleave2','finish4'] if bsz*q == 1 else ['interleave2','finish4','route2'] if bsz*q == 8 else ['serial16_32'])
    for seed in (7101,7102):
        c = dict(batch=bsz, ctx=80000, layer=layer, page=1, dist='soft', q_len=q, spread=('agentic' if bsz>1 else 'fixed'), seed=seed+10*i)
        d=R.generate_input(**c,local_start=R.case_fields(c,w)['local_start'])
        graphs={}
        for name in V:
            select(name)
            g,o=cap(d)
            ok,msg=R.check_implementation(d,R.collect(d,o),w)
            print('CHECK',name,c,ok,msg[:150],flush=True)
            if ok: graphs[name]=g
        ts={name:[] for name in graphs}
        for rep in range(24):
            for name in (list(graphs) if rep%2 else list(graphs)[::-1]): ts[name].append(t1(graphs[name]))
        med={name:statistics.median(t) for name,t in ts.items()}
        print('ROW',json.dumps(dict(case=c,times=med,gain={name:med['base']/t for name,t in med.items()})),flush=True)
        if seed==7101:
            for name in graphs:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graphs[name].replay();torch.cuda.synchronize()
                events=[(e.name,e.device_time_total) for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                print('PROFILE',json.dumps(dict(case=c,variant=name,kernels=events)),flush=True)
        del graphs
print('DONE',flush=True)
