"""Real sliding attention, all keys, cold-L2 captured graphs, R correctness."""
import triton
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
sys.path.insert(0,'/w/modal')
from compile_split import build
mod=build('muse_cluster_attention',candidate.CPP,candidate.CUDA,['cluster_attention'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])
def fused(data):
 c=data['config'];qkv,q=base._fd_prepare(data)
 chunk=triton.cdiv(triton.cdiv(c['window']+7,8),64)*64
 return mod.cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.)).view(c['batch'],8,-1)
candidate.kernel_fn=fused
for batch,ctx,page,scale,dist in ((1,6000,64,1.,'edges'),(4,9000,64,1/64,'soft'),(1,80000,1,1.,'coherent'),(4,80000,1,1.,'coherent')):
 for seed in (113,7919):
  data=R.generate_input(batch,ctx,'tp8','sliding',seed,q_len=8,page=page,kv_scale=scale,dist=dist,spread='agentic')
  gc,oc=capture(candidate,data)
  ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,ctx,seed,ok,msg,flush=True);assert ok,msg
  gb,ob=capture(base,data)
  bt,ct,ratios=[],[],[]
  for rep in range(24):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(batch=batch,ctx=ctx,page=page,seed=seed,baseline_us=statistics.median(bt),cluster_us=statistics.median(ct),speedup=statistics.median(ratios))),flush=True)
  del data,gc,gb,oc,ob
print('CLUSTER_ATTENTION_PASSED',flush=True)
