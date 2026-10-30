"""Sweep safe WGS grid sizes, comparing paired cold-L2 graph replays."""
import importlib.util, json, statistics, sys
import torch
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R
from checks import compare
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
print('START',flush=True)
w=R.generate_weights(R.WEIGHTS_SEED)
st=S.prepare(w)
sm_count=torch.cuda.get_device_properties('cuda').multi_processor_count
sfx=st.get('sfx', ())
ext=S._ext(tuple(S._main_defines(sfx))+sfx)
assert sm_count==st['grid'] and sm_count<=159
candidates=sorted(set([sm_count,sm_count-4,sm_count-12,sm_count-20]))
assert all(1<=n<=sm_count for n in candidates)
flush=torch.zeros(256*1024*1024//4,device='cuda',dtype=torch.int32)
print('GRID',json.dumps({'sm_count':sm_count,'candidates':candidates}),flush=True)

def capture(data, grid):
    x=data['hidden_states'].contiguous(); ids=data['topk_ids'].to(torch.int32).contiguous(); tw=data['topk_weights'].float().contiguous()
    out=torch.empty_like(x)
    def f(): ext.okc_run(st['w'],st['scratch'],x,ids,tw,out,0,32,grid)
    f(); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    g.replay(); torch.cuda.synchronize()
    return g,out,x,ids,tw

def timed(g):
    flush.add_(1); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000

for bs in (32,128,256):
    seed=419
    d=R.generate_input(bs,'ep8',seed)
    with torch.no_grad(): ref=R.ref_kernel(d,w)
    graphs={}
    for grid in candidates:
        graphs[grid]=capture(d,grid)
        ok,msg=compare(graphs[grid][1],ref,R.REL_L2_TOL)
        if not ok: raise AssertionError(f'grid={grid} bs={bs}: {msg}')
    base=graphs[sm_count][0]
    for grid in candidates:
        if grid==sm_count:continue
        tb=[];tc=[]
        for rep in range(20):
            if rep&1:tc.append(timed(graphs[grid][0]));tb.append(timed(base))
            else:tb.append(timed(base));tc.append(timed(graphs[grid][0]))
        print('ROW',json.dumps({'bs':bs,'seed':seed,'grid':grid,'base_grid':sm_count,
              'base_us':statistics.median(tb),'candidate_us':statistics.median(tc),
              'gain':statistics.median(tb)/statistics.median(tc)}),flush=True)
    del graphs,ref,d
print('DONE',flush=True)
