"""Cluster-size search. Every timing case receives the independent task check."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import triton
sys.path.insert(0,'/w/modal');from compile_split import build
mods={cs:build('muse_cluster_cs'+str(cs),candidate.CPP,getattr(candidate,'CUDA_CS'+str(cs)),['cluster_attention'],['-O3','-gencode=arch=compute_90a,code=sm_90a']) for cs in (1,2,4,8)}
def fused(data):
 c=data['config'];qkv,q=base._fd_prepare(data)
 chunk=triton.cdiv(triton.cdiv(c['window']+7,chosen),64)*64
 return mods[chosen].cluster_attention(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],qkv,c['page_size'],c['window'],chunk,c['sm_scale']*c['qk_scale_factor'],int(c['kv_scale']!=1.)).view(c['batch'],8,-1)
candidate.kernel_fn=fused
for batch in (1,16,32):
 for seed in (173,2387):
  data=R.generate_input(batch,80000,'tp8','sliding',seed,q_len=8,page=1,dist='coherent',spread='agentic')
  for chosen in (1,2,4,8):
   gc,oc=capture(candidate,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,seed,chosen,ok,msg,flush=True);assert ok,msg
   times=[elapsed(gc) for _ in range(24)]
   print('RESULT',json.dumps(dict(batch=batch,seed=seed,cluster_size=chosen,us=statistics.median(times))),flush=True)
   del gc,oc
  del data
print('CLUSTER_SIZES_PASSED',flush=True)
