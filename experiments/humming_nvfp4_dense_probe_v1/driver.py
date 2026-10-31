"""Isolated BF16 x NVFP4/E4M3 group-16 compatibility and timing probe."""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/w/humming')
sys.path.insert(0, '/w/kernels/muse_moe_w4a16')
import torch
import nvfp4
from humming.layer import HummingLayer

seed = 2371
torch.manual_seed(seed)
gen = torch.Generator(device='cuda').manual_seed(seed)
rows, k = 4096, 4096
packed, scales, gs = nvfp4.random_weights(rows, k, generator=gen, device='cuda')
layer = HummingLayer(shape_n=rows, shape_k=k,
    weight_config={'dtype':'float4e2m1','group_size':16,'scale_dtype':'float8e4m3','scale_2_type':'tensor'},
    input_config={'dtype':'bfloat16'}, torch_dtype=torch.bfloat16).to('cuda')
layer.load_from_tensors({'weight':packed.view(torch.int32), 'weight_scale':scales, 'weight_scale_2':gs.reshape(1)})
layer.transform()
print('LAYER_READY',torch.cuda.get_device_name(),flush=True)
w = nvfp4.dequantize(packed, scales, gs).to(torch.bfloat16)
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device='cuda')
def cap(fn):
    out=fn(); torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    g.replay(); torch.cuda.synchronize(); return g,out
def elapsed(g):
    flush.add_(1); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for m in (8,32,128,256):
    x=torch.randn((m,k),dtype=torch.bfloat16,device='cuda')
    x[0].zero_();x[0,17]=1
    gh,out=cap(lambda:layer(x)); gt,ref=cap(lambda:torch.mm(x,w.T))
    rel=float((out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
    row=float((out[0].float()-ref[0].float()).norm()/ref[0].float().norm().clamp_min(1e-12))
    finite=bool(torch.isfinite(out).all())
    if not finite or rel>0.01 or row>0.001: raise AssertionError((m,rel,row,finite))
    ht,tt,rat=[],[],[]
    for i in range(12):
        if i&1: t,h=elapsed(gt),elapsed(gh)
        else: h,t=elapsed(gh),elapsed(gt)
        ht.append(h);tt.append(t);rat.append(t/h)
    print('RESULT',json.dumps({'m':m,'humming_us':statistics.median(ht),'materialized_bf16_us':statistics.median(tt),'reference_over_humming':statistics.median(rat),'relative_l2':rel,'onehot_l2':row}),flush=True)
print('DONE',flush=True)
