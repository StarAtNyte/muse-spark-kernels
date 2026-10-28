"""Diagnostic ablations only. Incorrect ablated outputs are never candidates."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
import ast
sp=importlib.util.spec_from_file_location('bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
open('/w/run/frozen_parent.py','w').write(m.PARENT)
sp=importlib.util.spec_from_file_location('frozen_parent','/w/run/frozen_parent.py');sub=importlib.util.module_from_spec(sp);sys.modules['frozen_parent']=sub;sp.loader.exec_module(sub)
t=ast.parse(m.PARENT);n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.PARENT,n).replace('if tokens >= 32:','if False:'),sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('moe_ablate_v1',m.CPP,m.CUDA,['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf'],['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a'])
mod.wm_set_pdl_t(1);sub._wm_module=lambda:mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);wm=state['wm'];flush=torch.zeros(256*1024*1024//4,device='cuda')
def elapsed(g):
    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs in (128,256):
 for seed in (419,7919):
    data=R.generate_input(bs,'ep8',seed)
    result=sub.kernel_fn(data,state);ok,msg=R.check_implementation(data,result,weights)
    print('BASE_CHECK',bs,seed,ok,msg,flush=True);assert ok,msg
    dev='cuda';lmax=(bs+7)//8*8
    cnt=torch.empty(32,dtype=torch.int32,device=dev);lst=torch.empty(32*lmax,dtype=torch.int32,device=dev)
    act=torch.empty(512,dtype=torch.int32,device=dev);nact=torch.empty(1,dtype=torch.int32,device=dev)
    lids=torch.empty(bs*8,dtype=torch.int32,device=dev);woff=torch.empty(128,dtype=torch.int32,device=dev)
    mod.wm_route(data['topk_ids'],sub._WM_NOLS,0,lmax,cnt,lst,act,nact,lids,woff,32)
    maxslots=min(bs*8,32+bs*8//32)
    inter=torch.empty((bs*8,4096),dtype=torch.bfloat16,device=dev)
    def call(mode,dbg,out):
        p='13' if mode==0 else '2'
        mod.wm_gemm_run(mode,4,4,4,4,dbg,wm['w'+p],wm['s'+p],wm['gs'+p],data['hidden_states'] if mode==0 else inter,out,cnt,lst,act,nact,woff,maxslots,lmax)
    call(0,0,inter);torch.cuda.synchronize()
    for mode in (0,1):
        graphs={};outs={}
        for dbg in (0,1,2,3,32):
            outs[dbg]=torch.empty_like(inter);call(mode,dbg,outs[dbg]);g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call(mode,dbg,outs[dbg])
            g.replay();torch.cuda.synchronize();graphs[dbg]=g
        for dbg in (1,2,3,32):
            bt=[];ct=[]
            for rep in range(24):
                if rep%2:c,b=elapsed(graphs[dbg]),elapsed(graphs[0])
                else:b,c=elapsed(graphs[0]),elapsed(graphs[dbg])
                bt.append(b);ct.append(c)
            print('ABLATION',json.dumps(dict(batch=bs,seed=seed,mode=mode,dbg=dbg,baseline_us=statistics.median(bt),ablated_us=statistics.median(ct),ratio=statistics.median(bt)/statistics.median(ct))),flush=True)
print('DIAGNOSTIC_COMPLETE_NO_CANDIDATE',flush=True)
