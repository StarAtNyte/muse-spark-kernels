"""Paired WGS scale compression screen using the real standalone MoE path."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
B = load('base', '/w/kernels/muse_moe_w4a16/submission.py')
C = load('candidate', '/w/run/sub.py')
C.TINY_DEFINES = (); C.TINY2_DEFINES = (); C.MID_DEFINES = ()
old = """const uint16_t* sp = reinterpret_cast<const uint16_t*>(pc + 8192 + (wp * 32 + lane) * 6);
    const uint32_t sc0 = (uint32_t)sp[0] | ((uint32_t)sp[1] << 16), sc1 = (uint32_t)sp[2];"""
new = """const int scale_index = wp * 32 + lane;
    const int scale_offset = scale_index * 6;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(pc + 8192 + (scale_offset & ~3));
    const uint32_t sa = sp[0], sb = sp[1];
    const uint32_t sc0 = (scale_offset & 2) ? (sa >> 16) | (sb << 16) : sa;
    const uint32_t sc1 = (scale_offset & 2) ? (sb >> 16) : (sb & 0xffffu);"""
assert C.CUDA_SRC.count(new)==1
C.ALT_SRCS['uncompressed'] = C.CUDA_SRC.replace(new,old)
weights = R.generate_weights(5252)
sc = C.prepare(weights)
print('PREPARED CANDIDATE', flush=True)
sb = sc  # both compressed variants use identical weights and scratch
print('PREPARED PARENT',flush=True)
base_defines=tuple(C.MAIN_DEFINES); flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.int32)
def capture(d, state, base):
    C.MAIN_DEFINES=base_defines + (('@uncompressed',) if base else ())
    out=C.kernel_fn(d,state);torch.cuda.synchronize()
    ok,msg=R.check_implementation(d,out,weights)
    print('CHECK',base,ok,msg,flush=True)
    if not ok: raise AssertionError(msg)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=C.kernel_fn(d,state)
    graph.replay();torch.cuda.synchronize();return graph,out
def elapsed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs in (32,128,256):
    for seed in (419,7919):
        d=R.generate_input(bs,'ep8',seed);gb,ob=capture(d,sb,True);gc,oc=capture(d,sc,False)
        err=((ob.float()-oc.float()).norm()/ob.float().norm().clamp_min(1e-20)).item()
        if err>0.001: raise AssertionError(('relative difference',err))
        bt=[];ct=[]
        for i in range(20):
            if i&1:c,b=elapsed(gc),elapsed(gb)
            else:b,c=elapsed(gb),elapsed(gc)
            bt.append(b);ct.append(c)
        print('RESULT',json.dumps({'batch':bs,'seed':seed,'base_us':statistics.median(bt),'candidate_us':statistics.median(ct),'ratio':statistics.median(bt)/statistics.median(ct),'relative_l2':err}),flush=True)
print('DONE',flush=True)
