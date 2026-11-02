import sys
from unittest.mock import MagicMock
import torch
import torch.nn.functional as F
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import reference as R

def quant_group32(w):
    # Best symmetric int4 max-range fit, in the scale format expected by W4A16 kernels.
    orig = w.float()
    g = orig.reshape(*orig.shape[:-1], orig.shape[-1] // 32, 32)
    scale = (g.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-12).to(torch.bfloat16)
    q = torch.round(g / scale.float()).clamp(-8, 7)
    rec = (q * scale.float()).reshape_as(orig)
    return rec, (q != 0).float().mean().item(), ((rec-orig).square().mean()/orig.square().mean()).sqrt().item()

def run_quant(data, W13, W2, weights):
    x=data['hidden_states'].float(); ids=data['topk_ids'].long(); tw=data['topk_weights'].float()
    cfg=data['config']; lo=cfg['local_start']; hi=lo+cfg['local_count']
    out=torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi: continue
        tok,slot=(ids==e).nonzero(as_tuple=True)
        h=x[tok] @ W13[e-lo].T
        y=F.silu(h[:,:R.D_EXPERT])*h[:,R.D_EXPERT:]
        y=y @ W2[e-lo].T
        y=R.post_expert_norm(y, weights['post_norm_w'])
        out.index_add_(0,tok,y*tw[tok,slot,None])
    return out.to(torch.bfloat16)

weights=R.generate_weights(5252)
local_start=0; local_end=32
W13=torch.stack([R.dequantize(weights['w13'][e],weights['w13_sf'][e],weights['w13_gs'][e]) for e in range(local_start,local_end)])
W2=torch.stack([R.dequantize(weights['w2'][e],weights['w2_sf'][e],weights['w2_gs'][e]) for e in range(local_start,local_end)])
Q13,_,e13=quant_group32(W13); Q2,_,e2=quant_group32(W2)
print('WEIGHT_REL_L2',round(e13,6),round(e2,6),flush=True)
for bs,seed in ((32,20491),(128,20492),(256,20493)):
    data=R.generate_input(bs,'ep8',seed)
    # Keep the evaluated EP8 shard aligned with the exact local weights used above.
    data['config']['local_start']=local_start
    data['config']['local_count']=local_end-local_start
    ref=R.ref_kernel(data,weights)
    out=run_quant(data,Q13,Q2,weights)
    ok,msg=R.check_implementation(data,out,weights)
    diff=(out.float()-ref.float())
    rel=float(diff.norm()/ref.float().norm().clamp_min(1e-20))
    print('ROW',{'tokens':bs,'seed':seed,'passed':bool(ok),'rel_l2':round(rel,6),'message':msg[:180]},flush=True)
