"""M1 cooperative graph correctness, fresh routes and full-pipeline timings."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
sp=importlib.util.spec_from_file_location('bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
base_mod=build('muse_coop_base',m.BASE_CPP,m.BASE_CUDA,['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf','wm_permx'],flags)
base_mod.wm_set_pdl_t(1);sub._wm_module=lambda:base_mod
coop=build('muse_single_token_cooperative',m.CPP,m.CUDA,['cooperative_moe'],flags)
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);w=state['wm'];pids=torch.arange(8,device='cuda',dtype=torch.int32)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def fused(data,cap,prefetch):
 return coop.cooperative_moe(w['w13'],w['s13'],w['gs13'],w['w2'],w['s2'],w['gs2'],data['hidden_states'],data['topk_ids'],data['topk_weights'],state['norm_w'],pids,sub.POST_NORM_EPS,cap,prefetch)
def capture(fn,data):
 fn(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=fn(data)
 g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
 flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for seed in (113,419,2387,7919):
 data=R.generate_input(1,'ep8',seed)
 print('ROUTING',seed,data['topk_ids'].tolist(),flush=True)
 gb,ob=capture(lambda d:sub.kernel_fn(d,state),data)
 for cap,prefetch in ((512,0),(512,1),(264,0),(264,1),(132,0),(132,1)):
  gc,oc=capture(lambda d:fused(d,cap,prefetch),data)
  ok,msg=R.check_implementation(data,oc,weights);err=((oc.float()-ob.float()).norm()/ob.float().norm()).item()
  print('CHECK',seed,cap,prefetch,ok,msg,'base_error',err,flush=True);assert ok,msg
  bt,ct,ratios=[],[],[]
  for rep in range(24):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(seed=seed,cap=cap,prefetch=prefetch,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),error=err)),flush=True)
  if seed==113 and cap==512 and prefetch==0:
   saved=data['topk_ids'].clone()
   for ids in ([32]*8,[32]*7+[0],list(range(8))):
    data['topk_ids'].copy_(torch.tensor([ids],device='cuda',dtype=torch.int32))
    gc.replay();torch.cuda.synchronize()
    if all(e>=32 for e in ids):
     assert torch.isfinite(oc).all().item() and torch.count_nonzero(oc).item()==0
     ok,msg=True,'exact zero; no-local input is outside judge check domain'
    else:ok,msg=R.check_implementation(data,oc,weights)
    print('MUTATION_CHECK',ids,ok,msg,flush=True);assert ok,msg
   data['topk_ids'].copy_(saved)
  del gc,oc
 del gb,ob,data
print('MOE_COOPERATIVE_SCREEN_PASSED',flush=True)
