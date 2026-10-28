"""One compile, targeted pipeline/register study of dense transposed attention."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import types,time
from torch.utils.cpp_extension import load_inline
src=candidate._SP_CUDA.replace('template<int ST>','template<int ST,int LB>').replace('__launch_bounds__(128,2)','__launch_bounds__(128,LB)')
device,host=src.split('void sparse_decode_run(',1)
variants={'s2b1':(2,1),'s2b2':(2,2),'s2b3':(2,3),'s3b2':(3,2),'s4b2':(4,2)}
cpp=[];sources=[device];functions=[]
for name,(st,lb) in variants.items():
 fn='run_'+name
 code=('void sparse_decode_run('+host).replace('sparse_decode_run',fn).replace('2*8192',f'{st}*8192').replace('trans_decode<2>',f'trans_decode<{st},{lb}>')
 sources.append(code)
 cpp.append(candidate._SP_CPP.replace('sparse_decode_run',fn));functions.append(fn)
sources.append('std::vector<int64_t> trans_attrs(){std::vector<int64_t> v;cudaFuncAttributes a;')
for st,lb in variants.values():sources.append(f'cudaFuncGetAttributes(&a,trans_decode<{st},{lb}>);v.push_back(a.numRegs);v.push_back(a.localSizeBytes);')
sources.append('return v;}');cpp.append('std::vector<int64_t> trans_attrs();');functions.append('trans_attrs')
start=time.time();mod=load_inline('muse_transposed_tune', '\n'.join(cpp),cuda_sources='\n'.join(sources),functions=functions,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
print('COMPILE_SECONDS',time.time()-start,'RESOURCES',list(mod.trans_attrs()),flush=True)
sys.path.insert(0,'/w/modal');from compile_split import build
base._FD_MOD=build('muse_fd_transpose_tune_parent',base._FD_CPP,base._FD_CUDA,['fd_run_t','slide_run_t','pf_t','fd2_set_dbg_t','fd_set_pdl_t'],['-O3']+base._FD_FLAGS);base._FD_MOD.fd_set_pdl_t(1)
for layer,batch,options in (('sliding',16,(16,32)),('sliding',32,(8,16)),('full',16,(16,32)),('full',32,(16,64)),('full',1,(256,512))):
 for seed in (539,1427):
  data=R.generate_input(batch,80000,'tp8',layer,seed,page=1,spread='agentic',dist='soft' if layer=='sliding' else 'coherent')
  gb,ob=capture(base,data)
  for name in variants:
   candidate._SP_MOD=types.SimpleNamespace(sparse_decode_run=getattr(mod,'run_'+name))
   for splits in options:
    candidate._SP_SPLITS_OVERRIDE=splits;gc,oc=capture(candidate,data)
    err=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
    assert torch.isfinite(oc).all().item() and err<.003,(layer,batch,seed,name,splits,err)
    bt,ct,ratios=[],[],[]
    for rep in range(12):
     if rep%2:c,b=elapsed(gc),elapsed(gb)
     else:b,c=elapsed(gb),elapsed(gc)
     bt.append(b);ct.append(c);ratios.append(b/c)
    print('RESULT',json.dumps(dict(variant=name,layer=layer,batch=batch,seed=seed,splits=splits,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
    del gc,oc
  del gb,ob,data
print('TRANSPOSE_TUNE_PASSED',flush=True)
