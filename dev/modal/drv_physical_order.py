"""All sorting work is included in whole-pipeline timings.

Sort-only and presorted-attention timings are diagnostic, never submitted.
"""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import types
sys.path.insert(0,'/w/modal');from compile_split import build
mod=build('muse_physical_order',candidate.CPP,candidate.CUDA,['physical_sort'],['-O3','-gencode=arch=compute_90a,code=sm_90a'])

def check_table(original,seq,out):
 for b,n in enumerate(seq.tolist()):
  assert torch.equal(out[b,:n],original[b,:n].sort().values)
  assert torch.equal(out[b,n:],original[b,n:])

# Independently check order/permutation, causal append preservation and replay
# with changed table values and sequence lengths at fixed shapes.
for mode in (0,1):
 table=torch.full((3,257),-1,device='cuda',dtype=torch.int32)
 seq=torch.tensor([77,201,243],device='cuda',dtype=torch.int32)
 perm=torch.randperm(3*257,device='cuda',dtype=torch.int32).view(3,257)
 for b,n in enumerate(seq.tolist()):table[b,:n+8]=perm[b,:n+8]
 fn=types.SimpleNamespace(kernel_fn=lambda _:mod.physical_sort(table,seq,10,mode))
 graph,out=capture(fn,None);check_table(table,seq,out)
 seq.copy_(torch.tensor([119,39,229],device='cuda',dtype=torch.int32))
 table.fill_(-1);perm=torch.randperm(3*257,device='cuda',dtype=torch.int32).view(3,257)
 for b,n in enumerate(seq.tolist()):table[b,:n+8]=perm[b,:n+8]
 graph.replay();torch.cuda.synchronize();check_table(table,seq,out)
 print('METADATA_REPLAY_CHECK',mode,'PASS',flush=True)

def ordered(data,mode):
 bits=(data['k_cache'].shape[0]-1).bit_length()
 changed=dict(data);changed['page_table']=mod.physical_sort(data['page_table'],data['seq_lens'],bits,mode)
 return base.kernel_fn(changed)

for batch in (16,32):
 for seed in (173,2387):
  data=R.generate_input(batch,80000,'tp8','full',seed,q_len=8,page=1,spread='agentic',dist='coherent')
  gb,ob=capture(base,data)
  for mode in (0,1):
   wrapper=types.SimpleNamespace(kernel_fn=lambda d:ordered(d,mode));gc,oc=capture(wrapper,data)
   ok,msg=R.check_implementation(data,R.collect(data,oc));print('CHECK',batch,seed,mode,ok,msg,flush=True);assert ok,msg
   bt,ct,ratios=[],[],[]
   for rep in range(20):
    if rep%2:c,b=elapsed(gc),elapsed(gb)
    else:b,c=elapsed(gb),elapsed(gc)
    bt.append(b);ct.append(c);ratios.append(b/c)
   bits=(data['k_cache'].shape[0]-1).bit_length()
   sort_wrapper=types.SimpleNamespace(kernel_fn=lambda d:mod.physical_sort(d['page_table'],d['seq_lens'],bits,mode))
   gs,sorted_table=capture(sort_wrapper,data)
   changed=dict(data);changed['page_table']=sorted_table
   ga,oa=capture(base,changed)
   st,at=[],[]
   for _ in range(12):st.append(elapsed(gs));at.append(elapsed(ga))
   print('RESULT',json.dumps(dict(batch=batch,seed=seed,mode=mode,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),whole_sorted_us=statistics.median(ct),sort_only_us=statistics.median(st),presorted_attention_us=statistics.median(at))),flush=True)
   del gc,oc,gs,ga,oa,sorted_table,changed
  del gb,ob,data
print('PHYSICAL_ORDER_SCREEN_PASSED',flush=True)
