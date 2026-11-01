"""Exploratory grouped expert GEMMs vs the incumbent prefill MoE component."""
import importlib.util
import json
import statistics
import sys
from unittest.mock import MagicMock
import torch
import torch.nn.functional as F

class StubSGLang:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('sglang','sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=MagicMock();m.__path__=[];m.__name__=spec.name;m.__spec__=spec;return m
    def exec_module(self,module): pass
sys.meta_path.insert(0,StubSGLang())
sys.path[:0]=['/w/humming','/w/kernels/muse_prefill_layer']
import reference as R
from humming.layer import HummingLayer
from humming.config import GemmType
from humming.tune import get_heuristics_config
spec=importlib.util.spec_from_file_location('submission','/w/kernels/muse_prefill_layer/submission.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
weights=R.generate_weights(R.WEIGHTS_SEED)
state=S.prepare(weights); state['attn_choice']=dict(S.ATTN_FIXED)
print('PARENT_READY',flush=True)

def layer(n, prefix):
    L=HummingLayer(shape_n=n,shape_k=4096,num_experts=32,
        weight_config={'dtype':'float4e2m1','group_size':16,'scale_dtype':'float8e4m3','scale_2_type':'tensor'},
        input_config={'dtype':'bfloat16'},torch_dtype=torch.bfloat16).cuda()
    L.load_from_tensors({'weight':weights[prefix].view(torch.int32),
        'weight_scale':weights[prefix+'_sf'], 'weight_scale_2':weights[prefix+'_gs'].reshape(32)})
    L.transform()
    tune=get_heuristics_config(layer_config=L.humming_config,gemm_type=GemmType.GROUPED_CONTIGUOUS)
    return L,json.dumps(tune)

L1,T1=layer(8192,'w13'); L2,T2=layer(4096,'w2')
print('HUMMING_READY',flush=True)
# Counts from an archived local reference-checked run, used solely for an offline diagnostic.
counts=[0,107,15,65,0,0,1,0,2,39,222,0,0,5,0,0,111,0,0,1,472,0,0,0,0,0,0,2,0,0,2,0]
T=753
lid,pos,cnt=S._routes_from_counts(T,counts,'cuda')
routes=sum(counts)
index=torch.arange(routes,device='cuda',dtype=torch.long)
tokens=index % T
slots=index // T
p=torch.randn((T,4096),dtype=torch.bfloat16,device='cuda')
pf=p.float(); xn=(pf*torch.rsqrt(pf.square().mean(-1,keepdim=True)+1e-8)*(1+state['g_pre_expert_norm'].float())).to(torch.bfloat16)
xgroup=xn.index_select(0,tokens)
layout=torch.tensor([0]+list(__import__('itertools').accumulate(counts)),device='cuda',dtype=torch.int64)
config=json.dumps({'gemm_type':'grouped_contiguous'})
flush=torch.empty(256*1024*1024//4,device='cuda',dtype=torch.int32)

def humming():
    h=L1(xgroup,expert_layout=layout,compute_config=config,tuning_config=T1)
    inter=(F.silu(h[:,:4096].float())*h[:,4096:].float()).to(torch.bfloat16)
    return L2(inter,expert_layout=layout,compute_config=config,tuning_config=T2)
def parent(): return S._moe_forward(state,p,lid,pos,cnt,T)
def capture(fn):
    out=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
    flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000

gp,op=capture(parent)
for hint in (128, 256, 512, 753, 1044, 2048):
    try:
        cfg1=get_heuristics_config(layer_config=L1.humming_config,shape_m=hint,gemm_type=GemmType.GROUPED_CONTIGUOUS)
        cfg2=get_heuristics_config(layer_config=L2.humming_config,shape_m=hint,gemm_type=GemmType.GROUPED_CONTIGUOUS)
        t1=json.dumps(cfg1); t2=json.dumps(cfg2)
        def trial():
            h=L1(xgroup,expert_layout=layout,compute_config=config,tuning_config=t1)
            inter=(F.silu(h[:,:4096].float())*h[:,4096:].float()).to(torch.bfloat16)
            return L2(inter,expert_layout=layout,compute_config=config,tuning_config=t2)
        gc,oc=capture(trial)
        ref=op['y'][op['rrow'][tokens,slots].long()].float()
        err=float((oc.float()-ref).norm()/ref.norm().clamp_min(1e-12))
        if not torch.isfinite(oc).all() or err>0.05: raise AssertionError(err)
        bt=[];ct=[];rat=[]
        for i in range(12):
            if i&1: c,b=elapsed(gc),elapsed(gp)
            else: b,c=elapsed(gp),elapsed(gc)
            bt.append(b);ct.append(c);rat.append(b/c)
        print('RESULT',json.dumps({'hint':hint,'cfg1':cfg1,'cfg2':cfg2,'parent_us':statistics.median(bt),'candidate_us':statistics.median(ct),'parent_over_candidate':statistics.median(rat),'relative_l2':err}),flush=True)
    except Exception as e:
        print('ERROR',hint,repr(e),flush=True)
print('DONE',flush=True)
