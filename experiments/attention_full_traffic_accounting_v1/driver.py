"""Reconstruct exact length samples used by the attention reference's CUDA RNG."""
import json, math
import torch
for seed in (4319, 419, 7919, 22073):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    u=torch.rand((32,),generator=gen,device='cuda')
    lens=(80000*(1+0.075*(2*u-1))).long().clamp(1,math.ceil(80000*1.075)).cpu()
    rows=int(lens.sum())
    print('SAMPLE',json.dumps({'seed':seed,'lengths':lens.tolist(),'sum_context_rows':rows,
         'max_causal_rows':rows+32*8,'kv_bytes_old_context':rows*2*2*64,
         'page_index_bytes_old_context':rows*4}),flush=True)
print('DONE',torch.cuda.get_device_name(),flush=True)
