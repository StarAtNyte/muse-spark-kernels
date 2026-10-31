"""Statistics on exact public reference-generated BF16 weights; not a codec benchmark."""
import sys,json,math,torch
sys.path.insert(0,'/w/kernels/muse_bf16_proj')
import reference as R
for seed in (R.WEIGHTS_SEED,7919):
 weights=R.generate_weights(seed)
 for name in ('qkv_gate','o','pooler','unpooler'):
  w=weights[name]; bits=w.view(torch.int16).to(torch.int32)&65535
  exp=(bits>>7)&255;hist=torch.bincount(exp.reshape(-1),minlength=256)
  vals,idx=hist.topk(16); n=exp.numel(); total=hist[hist>0].double()/n
  entropy=float((-(total*total.log2()).sum()).item());lut=torch.zeros(256,device=w.device,dtype=torch.bool);lut[idx]=True
  outlier=~lut[exp.long()]; nmiss=int(outlier.sum().item()); fallback=int(outlier.view(-1,64).any(dim=1).sum().item()); rows=n//64
  print('STATS',json.dumps(dict(seed=seed,layer=name,shape=list(w.shape),weights=n,entropy_bits=entropy,top16_fraction=float(vals.sum().item()/n),escape_count=nmiss,tile64_rows=rows,verbatim_rows=fallback,row_fraction=fallback/rows,palette=idx.cpu().tolist())),flush=True)
  del bits,exp,hist,total,lut,outlier
 del weights
print('DONE',flush=True)
