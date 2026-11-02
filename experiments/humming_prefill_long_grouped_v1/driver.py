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
spec=importlib.util.spec_from_file_location('submission','/w/run/sub.py')
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
counts=[0,1792,835,1835,0,0,1,0,7,445,4629,0,0,161,18,0,1951,0,0,19,11793,1,0,0,0,0,0,1,0,0,9,0]
T=16384
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

gp,op=capture(parent); gh,oh=capture(humming)
ref=op['y'][op['rrow'][tokens,slots].long()].float()
err=float((oh.float()-ref).norm()/ref.norm().clamp_min(1e-12))
finite=bool(torch.isfinite(oh).all())
print('COMPARE',json.dumps({'relative_l2_vs_parent':err,'finite':finite,'routes':routes}),flush=True)
if not finite or err>0.05: raise AssertionError(('output mismatch',err,finite))
pt=[];ht=[];rat=[]
for i in range(8):
    if i&1: h,b=elapsed(gh),elapsed(gp)
    else: b,h=elapsed(gp),elapsed(gh)
    pt.append(b);ht.append(h);rat.append(b/h)
print('RESULT',json.dumps({'parent_with_gather_us':statistics.median(pt),'humming_pregathered_us':statistics.median(ht),'parent_over_humming':statistics.median(rat),'relative_l2_vs_parent':err}),flush=True)
h=L1(xgroup,expert_layout=layout,compute_config=config,tuning_config=T1)
def activation(): return (F.silu(h[:,:4096].float())*h[:,4096:].float()).to(torch.bfloat16)
inter=activation()
stages=[('gemm1',lambda:L1(xgroup,expert_layout=layout,compute_config=config,tuning_config=T1)),
        ('silu_mul',activation),
        ('gemm2',lambda:L2(inter,expert_layout=layout,compute_config=config,tuning_config=T2))]
for name,fn in stages:
    g,out=capture(fn);times=[]
    for i in range(8): times.append(elapsed(g))
    print('STAGE',json.dumps({'name':name,'median_us':statistics.median(times),'min_us':min(times),'max_us':max(times)}),flush=True)
print('DONE',flush=True)
