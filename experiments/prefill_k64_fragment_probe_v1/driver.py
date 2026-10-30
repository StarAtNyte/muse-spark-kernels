"""One-stage-layout K64 permutation probe using real prefill moe_gemm WGMMA.

Only ungated GEMM is exercised; this is not a full MoE or judge candidate.
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
w,sf,gs=random_weights(128,4096,generator=gen,device='cuda',leading=(1,))
x=torch.randn((32,4096),device='cuda',dtype=torch.bfloat16)
perm=torch.tensor([16*((c>>1)&3)+4*j+(c&1)+2*(c>>3) for j in range(4) for c in range(16)],device='cuda')
assert torch.equal(torch.sort(perm).values,torch.arange(64,device='cuda'))
codes=S._codes(w).view(1,128,64,64).index_select(-1,perm).reshape(1,128,2048,2)
wp=(codes[...,0] | (codes[...,1]<<4)).contiguous()
wb=S._repack(w,sf,False);wc=S._repack(wp,sf,False)
def tile(a):
    # [stage, token, 8 chunks, 8 bf16], xor-swizzle 16-byte chunks by token row mod 8.
    rows=a.view(32,64,8,8).permute(1,0,2,3).contiguous()
    out=torch.empty_like(rows)
    for r in range(32):
        for ch in range(8): out[:,r,ch ^ (r&7),:]=rows[:,r,ch,:]
    return out.contiguous()
xb=tile(x);xp=tile(x.view(32,64,64).index_select(-1,perm).reshape(32,4096))
tiles=torch.tensor([[0,0,32,0]],device='cuda',dtype=torch.int32); nt=torch.tensor([1],device='cuda',dtype=torch.int32)
yb=torch.empty((32,4096),device='cuda',dtype=torch.bfloat16);yc=torch.empty_like(yb)
for mod,weight,inp,out in [(base,wb,xb,yb),(cand,wc,xp,yc)]:
    mod.moe_gemm(weight,inp,tiles,nt,gs.reshape(-1),out,1,1,False,smem)
    torch.cuda.synchronize()
ref=x.float() @ dequantize(w[0],sf[0],gs[0]).t()
for name,y in [('baseline',yb),('permuted',yc)]:
    out=y[:,:128].float(); err=((out-ref).norm()/ref.norm()).item(); maxerr=(out-ref).abs().max().item()
    print('RESULT',json.dumps({'name':name,'relative_l2':err,'max_abs':maxerr}),flush=True)
    if not torch.isfinite(out).all() or err>0.015:raise AssertionError((name,err))
print('PASS: real ungated WGMMA single-tile permutation and scale selection',flush=True)
