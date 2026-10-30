"""One-stage-layout K64 permutation probe using real prefill moe_gemm WGMMA.

Tests gated GEMM output order; this is not a full MoE or judge candidate.
"""
import importlib.util, sys, json
import torch
sys.path.insert(0,'/w/kernels/muse_prefill_layer')
from nvfp4 import random_weights, dequantize
spec=importlib.util.spec_from_file_location('sub','/w/run/sub.py')
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
old='scale_pair_sel(sw, j & 1, m16)'
new='scale_pair_sel(t < 2 ? sc.x : sc.y, t & 1, m16)'
assert S._CUDA_SRC.count(old)==4
base=S._build(True)
S._CUDA_SRC=S._CUDA_SRC.replace(old,new,1)
cand=S._build(True)
smem=S.MOE_SMEM_SM90
base.moe_init(smem);cand.moe_init(smem)
torch.manual_seed(921)
gen=torch.Generator(device='cuda').manual_seed(921)
w,sf,gs=random_weights(8192,4096,generator=gen,device='cuda',leading=(1,))
x=torch.randn((32,4096),device='cuda',dtype=torch.bfloat16)
perm=torch.tensor([16*((c>>1)&3)+4*j+(c&1)+2*(c>>3) for j in range(4) for c in range(16)],device='cuda')
assert torch.equal(torch.sort(perm).values,torch.arange(64,device='cuda'))
# Reorder the gate and up outputs identically so phase A publishes the K64 order for phase B.
row=(torch.arange(4096,device='cuda').view(64,64) // 64 * 64 + perm).reshape(-1)
rows=torch.cat((row,row+4096))
wr=w.index_select(1,rows); sfr=sf.index_select(1,rows)
codes=S._codes(wr).view(1,8192,64,64).index_select(-1,perm).reshape(1,8192,2048,2)
wp=(codes[...,0] | (codes[...,1]<<4)).contiguous()
wb=S._repack(w,sf,True);wc=S._repack(wp,sfr,True)
def tile(a):
    rows=a.view(32,64,8,8).permute(1,0,2,3).contiguous()
    out=torch.empty_like(rows)
    for r in range(32):
        for ch in range(8):out[:,r,ch^(r&7),:]=rows[:,r,ch,:]
    return out.contiguous()
xb=tile(x);xp=tile(x.view(32,64,64).index_select(-1,perm).reshape(32,4096))
tiles=torch.tensor([[0,0,32,0]],device='cuda',dtype=torch.int32);nt=torch.tensor([1],device='cuda',dtype=torch.int32)
yb=torch.empty((32,4096),device='cuda',dtype=torch.bfloat16);yc=torch.empty_like(yb)
# NT is physical number of 128-row gated tiles per expert, not the number launched: use NT=64.
# Run all output tiles so wrep's expert stride and writes match production.
for mod,weight,inp,out in [(base,wb,xb,yb),(cand,wc,xp,yc)]:
    mod.moe_gemm(weight,inp,tiles,nt,gs.reshape(-1),out,64,1,True,smem)
    torch.cuda.synchronize()
def untile(a):
    v=a.view(64,32,8,8); out=torch.empty_like(v)
    for r in range(32):
        for ch in range(8):out[:,r,ch,:]=v[:,r,ch^(r&7),:]
    return out.permute(1,0,2,3).reshape(32,4096)
b=untile(yb);c=untile(yc)
refw=dequantize(w[0],sf[0],gs)
refgate=x.float() @ refw[:4096].t();refup=x.float() @ refw[4096:].t()
ref=torch.nn.functional.silu(refgate)*refup
for name,out,expect in [('baseline',b,ref),('permuted',c,ref.index_select(1,row))]:
    err=((out.float()-expect).norm()/expect.norm()).item(); print('RESULT',json.dumps({'name':name,'relative_l2':err}),flush=True)
    if not torch.isfinite(out).all() or err>0.03:raise AssertionError((name,err))
err=(c.float()-b.index_select(1,row).float()).norm()/b.float().norm()
print('PERMUTED_VS_BASE',float(err),flush=True)
if err>0.01:raise AssertionError(('baseline_diff',float(err)))
print('PASS: gated WGMMA publishes K64-permuted intermediates',flush=True)
