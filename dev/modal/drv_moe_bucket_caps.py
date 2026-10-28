"""Exact count tiers; parallel streams join before finish and timing events."""
exec(open('/w/modal/drv_moe_preload_adaptive.py').read().split('# Generate the same raw format')[0])
sp=importlib.util.spec_from_file_location('bundle','/w/run/sub.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
import types,ast
open('/w/run/bucket_parent.py','w').write(m.BASE_SUBMISSION)
parent_spec=importlib.util.spec_from_file_location('bucket_frozen_parent','/w/run/bucket_parent.py')
sub=importlib.util.module_from_spec(parent_spec);sys.modules['bucket_frozen_parent']=sub;parent_spec.loader.exec_module(sub)
finish=next(n for n in ast.parse(m.BASE_SUBMISSION).body if isinstance(n,ast.FunctionDef) and n.name=='_finish_experts')
exec(ast.get_source_segment(m.BASE_SUBMISSION,finish).replace('if tokens >= 32:', 'if False:'),sub.__dict__)
sys.path.insert(0,'/w/modal');from compile_split import build
flags=['-O3','--use_fast_math','-gencode=arch=compute_90a,code=sm_90a']
functions=['wm_route','wm_gemm_run','wm_set_pdl_t','wm_set_persist_t','wm_finish','wm_pf','wm_permx']
base_mod=build('muse_bucket_base',m.BASE_CPP,m.BASE_CUDA,functions,flags)
bucket_mod=build('muse_count_bucket',m.CPP,m.CUDA,functions+['wm_bucket_run'],flags)
short_mod=build('muse_bucket_short',m.CPP,m.CUDA_SHORT,functions+['wm_bucket_run'],flags)
for mod in (base_mod,bucket_mod,short_mod):mod.wm_set_pdl_t(1)
sub._wm_module=lambda:base_mod
R.N_EXPERTS=32;weights=R.generate_weights(93819);R.N_EXPERTS=256
sub._prepare_marlin=lambda w:{'norm_w':(1+w['post_norm_w'].float()).bfloat16()}
state=sub.prepare(weights);exec(m.FORWARD,sub.__dict__)
state['_bucket_streams']=[torch.cuda.Stream() for _ in range(3)]
sms=torch.cuda.get_device_properties(0).multi_processor_count
state['_bucket_caps']=(sms*4,sms*2,sms*2)
flush=torch.zeros(256*1024*1024//4,device='cuda')
def capture(data,variant):
 state['wm']['mod']=base_mod if variant=='base' else short_mod if variant.startswith('short') else bucket_mod
 if variant!='base':
  c1,c2={'row24':(4,2),'row44':(4,4),'row54':(5,4),'short44':(4,4),'short54':(5,4)}[variant]
  state['_bucket_caps']=(sms*c1,sms*c2,sms*2)
 state['_bucket_parallel']=True
 fn=sub.kernel_fn if variant=='base' else sub._bucket_forward
 fn(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=fn(data,state)
 g.replay();torch.cuda.synchronize();return g,out
def elapsed(g):
 flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
def check_metadata(data):
 from collections import Counter
 bs=data['hidden_states'].shape[0];lmax=(bs+7)//8*8
 cnt=torch.empty(32,device='cuda',dtype=torch.int32);lst=torch.empty(32*lmax,device='cuda',dtype=torch.int32)
 act=torch.empty(2048,device='cuda',dtype=torch.int32);nact=torch.empty(4,device='cuda',dtype=torch.int32)
 lids=torch.empty(bs*8,device='cuda',dtype=torch.int32);woff=torch.empty(128,device='cuda',dtype=torch.int32)
 bucket_mod.wm_route(data['topk_ids'],sub._WM_NOLS,0,lmax,cnt,lst,act,nact,lids,woff,32)
 counts=nact.tolist();items=act.view(4,128,4).tolist();original=[tuple(v) for v in items[0][:counts[0]]]
 collected=[]
 for tier in range(3):
  for v in items[tier+1][:counts[tier+1]]:
   assert (0<v[2]<=8) if tier==0 else (8<v[2]<=16) if tier==1 else (16<v[2]<=32)
   collected.append(tuple(v))
 assert Counter(original)==Counter(collected)
 print('METADATA_CHECK',bs,counts,'PASS',flush=True)
for batch,dist,seed in [(128,'normal',419),(256,'normal',419),(128,'normal',7919),(256,'normal',7919),(128,'skewed',1487)]:
 data=R.generate_input(batch,'ep8',seed,dist=dist);check_metadata(data)
 gb,ob=capture(data,'base')
 for variant in ('row24','row44','row54','short44','short54'):
  gc,oc=capture(data,variant)
  ok,msg=R.check_implementation(data,oc,weights);err=((oc.float()-ob.float()).norm()/ob.float().norm()).item()
  print('CHECK',variant,batch,dist,seed,ok,msg,'base_error',err,flush=True);assert ok,msg
  bt,ct,ratios=[],[],[]
  for rep in range(20):
   if rep%2:c,b=elapsed(gc),elapsed(gb)
   else:b,c=elapsed(gb),elapsed(gc)
   bt.append(b);ct.append(c);ratios.append(b/c)
  print('RESULT',json.dumps(dict(variant=variant,batch=batch,dist=dist,seed=seed,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
  del gc,oc
 del gb,ob,data
print('MOE_BUCKET_CAPS_SCREEN_PASSED',flush=True)
