"""Paired MID (split-job skinny) vs MAIN (WGS) build on KPERM3 source at M=16/24/32."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('START',flush=True)
w=R.generate_weights(R.WEIGHTS_SEED); st=S.prepare(w)
sfx=st.get('sfx', ())
exts={'main':S._ext(tuple(S._main_defines(sfx))+sfx),'mid':S._ext(tuple(S.MID_DEFINES)+sfx)}
flush=torch.zeros(256*1024*1024//4,device='cuda',dtype=torch.int32)
def capture(data,ext):
    x=data['hidden_states'].contiguous(); ids=data['topk_ids'].to(torch.int32).contiguous(); tw=data['topk_weights'].float().contiguous()
    out=torch.empty_like(x)
    def f(): ext.okc_run(st['w'],st['scratch'],x,ids,tw,out,st['lo'],st['cnt'],st['grid'])
    f();torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):f()
    g.replay();torch.cuda.synchronize();return g,out,x,ids,tw
def timed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs,seeds in ((32,(3357109366520139634,419,7919)),(16,(3357068220476696277,))):
  for seed in seeds:
    data=R.generate_input(bs,'ep8',seed)
    with torch.no_grad():ref=R.ref_kernel(data,w)
    graphs={}
    for k,e in exts.items():
        graphs[k]=capture(data,e)
        ok,msg=compare(graphs[k][1],ref,R.REL_L2_TOL)
        if not ok:print('FAIL',k,bs,msg,flush=True)
    tb=[];tc=[]
    for i in range(40):
        a,b=(('main','mid') if i&1 else ('mid','main'))
        r={a:timed(graphs[a][0]),b:timed(graphs[b][0])}
        tb.append(r['main']);tc.append(r['mid'])
    print('ROW',json.dumps({'bs':bs,'seed':seed,'main_us':statistics.median(tb),'mid_us':statistics.median(tc),'gain_mid':statistics.median(tb)/statistics.median(tc)}),flush=True)
    del graphs,ref,data
print('DONE',flush=True)
