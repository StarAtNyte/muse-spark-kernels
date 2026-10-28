"""Real routed M128/256 cases; fixed weights, changing routes; cold-L2 pairs."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
sp=importlib.util.spec_from_file_location('bundle','/w/run/sub.py');bundle=importlib.util.module_from_spec(sp);sp.loader.exec_module(bundle)
sys.path.insert(0,'/w/modal');from compile_split import build
functions=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf','wm_permx']
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
mods={}
for name,cuda in [('base',bundle.CUDA_BASE),('control',bundle.CUDA_CONTROL),('stream',bundle.CUDA_STREAM),('paired',bundle.CUDA_PAIRED)]:
 mods[name]=build('muse_reuse64_'+name,bundle.CPP,cuda,functions,flags);mods[name].wm_set_pdl_t(1);print('COMPILED',name,flush=True)
sub._wm_module=lambda:mods['base']
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights)
base_fn=sub.kernel_fn
code=inspect.getsource(base_fn).replace('def kernel_fn(', 'def reuse_fn(')
code=code.replace('nch = 1 if m <= 8 else 2 if m <= 16 else 4','nch = 8')
code=code.replace('    nt1 = _c[3] if len(_c) > 3 else nt','    nt,nw,ring=2,4,9\n    nt1=2')
exec(code,sub.__dict__);reuse_fn=sub.reuse_fn
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,name):
 state['wm']['mod']=mods[name];fn=base_fn if name=='base' else reuse_fn
 fn(data,state);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=fn(data,state)
 graph.replay();torch.cuda.synchronize();return graph,out
def elapsed(graph):
 flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
for batch,dist,seed in [(128,'normal',419),(256,'normal',419),(128,'normal',7919),(256,'normal',7919),(128,'skewed',1487)]:
 data=R.generate_input(batch,'ep8',seed,dist=dist)
 ids=data['topk_ids'];counts=torch.bincount(ids[(ids>=0)&(ids<32)].long(),minlength=32)
 print('ROUTING',batch,dist,seed,counts.tolist(),flush=True)
 gb,ob=capture(data,'base')
 for name in ('control','stream','paired'):
  gc,oc=capture(data,name)
  error=((oc.float()-ob.float()).norm()/ob.float().norm()).item()
  ok,msg=R.check_implementation(data,oc,weights);print('CHECK',name,batch,dist,seed,ok,msg,'base_error',error,flush=True);assert ok,msg
  bt,ct,ratios=[],[],[]
  for rep in range(20):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(variant=name,batch=batch,dist=dist,seed=seed,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),error=error)),flush=True)
  del gc,oc
 del gb,ob,data
print('MOE_REUSE64_SCREEN_PASSED',flush=True)
