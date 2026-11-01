"""Paired MAIN vs MAIN+OKC_DRYW=1 through the file's own kernel_fn (cold L2 via read flush)."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('START',flush=True)
w=R.generate_weights(R.WEIGHTS_SEED); st=S.prepare(w)
BASE=tuple(S.MAIN_DEFINES); DRY=BASE+('-DOKC_DRYW=1',)
flush=torch.ones(256*1024*1024//4,device='cuda')
def capture(data,defs):
    S.MAIN_DEFINES=defs
    f=lambda: S.kernel_fn(data,st)
    f();torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=f()
    g.replay();torch.cuda.synchronize();return g,out
def timed(g):
    flush.sum();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs,seeds in ((32,(3357109366520139634,419,7919)),(16,(3357068220476696277,)),(128,(11,)),(64,(5,))):
  for seed in seeds:
    data=R.generate_input(bs,'ep8',seed)
    with torch.no_grad():ref=R.ref_kernel(data,w)
    gs={}
    for k,d in (('main',BASE),('dry',DRY)):
        gs[k]=capture(data,d)
        ok,msg=compare(gs[k][1],ref,R.REL_L2_TOL)
        if not ok:print('FAIL',k,bs,msg,flush=True)
    tb=[];tc=[]
    for i in range(40):
        a,b=(('main','dry') if i&1 else ('dry','main'))
        r={a:timed(gs[a][0]),b:timed(gs[b][0])}
        tb.append(r['main']);tc.append(r['dry'])
    print('ROW',json.dumps({'bs':bs,'seed':seed,'main_us':statistics.median(tb),'dry_us':statistics.median(tc),'gain':statistics.median(tb)/statistics.median(tc)}),flush=True)
    del gs,ref,data
print('DONE',flush=True)
