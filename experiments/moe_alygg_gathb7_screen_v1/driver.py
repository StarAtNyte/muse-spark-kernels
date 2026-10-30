"""Paired current KPERM3 WGS against GATHB=7 with common weights/inputs."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
S.TINY_DEFINES=(); S.TINY2_DEFINES=(); S.MID_DEFINES=()
old='#define OKC_GATHB 8'; assert S.CUDA_SRC.count(old)==1
S.ALT_SRCS['gath7']=S.CUDA_SRC.replace(old,'#define OKC_GATHB 7')
print('START',flush=True)
w=R.generate_weights(R.WEIGHTS_SEED); st=S.prepare(w)
sfx=st.get('sfx', ()); defs=tuple(S._main_defines(sfx))+sfx
exts={'base':S._ext(defs),'gath7':S._ext(defs+('@gath7',))}
flush=torch.zeros(256*1024*1024//4,device='cuda',dtype=torch.int32)

def capture(data,ext):
    x=data['hidden_states'].contiguous(); ids=data['topk_ids'].to(torch.int32).contiguous(); tw=data['topk_weights'].float().contiguous()
    out=torch.empty_like(x)
    def f(): ext.okc_run(st['w'],st['scratch'],x,ids,tw,out,0,32,st['grid'])
    f();torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):f()
    g.replay();torch.cuda.synchronize();return g,out,x,ids,tw

def timed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000

for bs in (32,128,256):
    seed=419;data=R.generate_input(bs,'ep8',seed)
    with torch.no_grad():ref=R.ref_kernel(data,w)
    graphs={}
    for k,e in exts.items():
        graphs[k]=capture(data,e)
        ok,msg=compare(graphs[k][1],ref,R.REL_L2_TOL)
        if not ok:raise AssertionError(f'{k} bs={bs}: {msg}')
    tb=[];tc=[]
    for i in range(20):
        if i&1:tc.append(timed(graphs['gath7'][0]));tb.append(timed(graphs['base'][0]))
        else:tb.append(timed(graphs['base'][0]));tc.append(timed(graphs['gath7'][0]))
    print('ROW',json.dumps({'bs':bs,'seed':seed,'base_us':statistics.median(tb),'candidate_us':statistics.median(tc),
                            'gain':statistics.median(tb)/statistics.median(tc)}),flush=True)
    del graphs,ref,data
print('DONE',flush=True)
