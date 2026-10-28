import importlib.util, json, statistics, sys
from unittest.mock import MagicMock
import torch, triton
class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,m): pass
sys.meta_path.insert(0,StubSGLang())
sys.path.insert(0,'/w/kernels/muse_moe_w4a16')
import reference as R

def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load('decoder_base','/w/kernels/muse_decode_layer/submission.py')
cand=load('decoder_candidate','/w/run/sub.py')
# The modal diagnostic image lacks SGLang's fused norm wrapper; execute the same
# Triton kernel directly so both expert backends include identical post-norm work.
def local_finish(mod, expert_output, topk_scores, norm_weight=None, eps=1e-8, weight_first=True, apply_norm=True, slot_expert_ids=None):
    tokens,topk,hidden=expert_output.shape
    out=torch.empty((tokens,hidden),device=expert_output.device,dtype=expert_output.dtype)
    mod._post_expert_norm_weighted_sum_kernel[(tokens,)](expert_output,norm_weight if norm_weight is not None else expert_output,topk_scores,out,slot_expert_ids if slot_expert_ids is not None else expert_output,TOPK=topk,H=hidden,eps=eps,APPLY_NORM=apply_norm,HAS_WEIGHT=norm_weight is not None,WEIGHT_FIRST=weight_first,HAS_IDS=slot_expert_ids is not None,BLOCK_H=triton.next_power_of_2(hidden),num_warps=8,num_stages=2)
    return out
base._finish_experts=lambda *a,**kw:local_finish(base,*a,**kw)
cand._finish_experts=lambda *a,**kw:local_finish(cand,*a,**kw)
layer_spec=importlib.util.spec_from_file_location('layer_reference','/w/kernels/muse_decode_layer/reference.py')
LayerR=importlib.util.module_from_spec(layer_spec);layer_spec.loader.exec_module(LayerR)
weights=LayerR.generate_weights(1234)
base_state={'wm':base._wm_prepare(weights),'post_expert_w':(weights['post_expert_norm'].float()+1).to(torch.bfloat16)}
omw=dict(weights);omw['post_norm_w']=weights['post_expert_norm']
cand_state={'wm':cand._wm_prepare(weights),'post_expert_w':base_state['post_expert_w'],'operatorstack_m32':cand._om_prepare_m32(omw)}
print('READY',torch.cuda.get_device_name(),flush=True)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def base_run(x,tw,ids,st): return base._before_stack_bucket_experts_wm(x,tw,ids,st)
def cand_run(x,tw,ids,st): return cand._experts_wm(x,tw,ids,st)
def cap(fn):
    y=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    g.replay();torch.cuda.synchronize();return g,o
def elapsed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for bs in (32,128):
 for seed in (419,7919,15241):
    d=R.generate_input(bs,'ep8',seed)
    x=d['hidden_states'];tw=d['topk_weights'].to(torch.float32).contiguous();ids=d['topk_ids'].to(torch.int32).contiguous()
    gb,ob=cap(lambda:base_run(x,tw,ids,base_state))
    gc,oc=cap(lambda:cand_run(x,tw,ids,cand_state))
    mw={'w13':weights['w13'],'w13_sf':weights['w13_sf'],'w13_gs':weights['w13_gs'],'w2':weights['w2'],'w2_sf':weights['w2_sf'],'w2_gs':weights['w2_gs'],'post_norm_w':weights['post_expert_norm']}
    ok,msg=R.check_implementation(d,oc,mw)
    err=((oc.float()-ob.float()).norm()/ob.float().norm().clamp_min(1e-20)).item()
    print('CHECK',bs,seed,ok,msg,'relative_to_decoder_winner',err,flush=True)
    if not ok or err>.015:raise RuntimeError('correctness threshold failed')
    bt=[];ct=[]
    for rep in range(20):
        if rep%2:ct.append(elapsed(gc));bt.append(elapsed(gb))
        else:bt.append(elapsed(gb));ct.append(elapsed(gc))
    print('RESULT',json.dumps({'batch':bs,'seed':seed,'decoder_us':statistics.median(bt),'hybrid_us':statistics.median(ct),'ratio':statistics.median(bt)/statistics.median(ct)}),flush=True)
print('DECODER_MOE_HYBRID_SCREEN_PASSED',flush=True)
