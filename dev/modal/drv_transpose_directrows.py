"""Compare larger tiles and conditional rescaling against the validated parent."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
import types,time
from torch.utils.cpp_extension import load_inline
helper=candidate._SP_CUDA.split('template<int ST>')[0]
sources=[helper];cpp=[];functions=[];variants={}
for name in ('stable','direct'):
 variants[name]=None
 src='template<int ST>'+candidate._SP_CUDA.split('template<int ST>',1)[1]
 if name=='direct':
  src=src.replace(' int* rows=reinterpret_cast<int*>(mem+ST*8192);','')
  src=src.replace(' for(int i=tid;i<n;i+=128){int tok=begin+i,pg=tok>>shift,phys=pg<table_n?TABLE[b*table_n+pg]:-1;rows[i]=phys<0?0:(phys<<shift)+(tok&mask);}','')
  src=src.replace('long rr=li<n?rows[li]:0;', 'int tok=begin+li,pg=tok>>shift,phys=(li<n&&pg<table_n)?TABLE[b*table_n+pg]:-1;long rr=phys<0?0:(phys<<shift)+(tok&mask);')
  src=src.replace('3*8192+((chunk+127)/128)*128*4','3*8192')
  assert 'rows[' not in src
 src=src.replace('trans_decode',f'trans_{name}').replace('sparse_decode_run',f'run_{name}')
 sources.append(src);cpp.append(candidate._SP_CPP.replace('sparse_decode_run',f'run_{name}'));functions.append(f'run_{name}')
sources.append('std::vector<int64_t> attrs(){std::vector<int64_t> v;cudaFuncAttributes a;')
for name in variants:sources.append(f'cudaFuncGetAttributes(&a,trans_{name}<3>);v.push_back(a.numRegs);v.push_back(a.localSizeBytes);')
sources.append('return v;}');cpp.append('std::vector<int64_t> attrs();');functions.append('attrs')
start=time.time();mod=load_inline('muse_transposed_directrows', '\n'.join(cpp),cuda_sources='\n'.join(sources),functions=functions,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
print('COMPILE_SECONDS',time.time()-start,'RESOURCES',list(mod.attrs()),flush=True)
sys.path.insert(0,'/w/modal');from compile_split import build
base._FD_MOD=build('muse_fd_transpose_directrows_parent',base._FD_CPP,base._FD_CUDA,['fd_run_t','slide_run_t','pf_t','fd2_set_dbg_t','fd_set_pdl_t'],['-O3']+base._FD_FLAGS);base._FD_MOD.fd_set_pdl_t(1)
for layer,batch,options in (('full',16,(32,)),('full',32,(16,)),('sliding',16,(16,)),('sliding',32,(8,))):
 for seed in (7919,3613,19231):
  data=R.generate_input(batch,80000,'tp8',layer,seed,page=1,spread='agentic',dist='soft' if layer=='sliding' else 'coherent')
  gb,ob=capture(base,data)
  for name in variants:
   candidate._SP_MOD=types.SimpleNamespace(sparse_decode_run=getattr(mod,'run_'+name))
   for splits in options:
    gc,oc=capture(candidate,data)
    err=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
    assert torch.isfinite(oc).all().item() and err<.003,(layer,batch,seed,name,splits,err)
    bt,ct,ratios=[],[],[]
    for rep in range(16):
     if rep%2:c,b=elapsed(gc),elapsed(gb)
     else:b,c=elapsed(gb),elapsed(gc)
     bt.append(b);ct.append(c);ratios.append(b/c)
    print('RESULT',json.dumps(dict(variant=name,layer=layer,batch=batch,seed=seed,splits=splits,error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),candidate_us=statistics.median(ct))),flush=True)
    del gc,oc
  del gb,ob,data
# Validate nearby shapes, scaled cache and paged cache on every variant.
for name in variants:
 candidate._SP_MOD=types.SimpleNamespace(sparse_decode_run=getattr(mod,'run_'+name))
 for batch,ctx,layer,page,scale,dist in ((17,9000,'full',64,1/64,'soft'),(33,6000,'sliding',64,1/64,'edges')):
  data=R.generate_input(batch,ctx,'tp8',layer,241+batch,page=page,kv_scale=scale,dist=dist)
  out=candidate.kernel_fn(data);ok,message=R.check_implementation(data,R.collect(data,out));print('CHECK',name,batch,layer,ok,message,flush=True);assert ok,message
print('TRANSPOSE_DIRECT_ROWS_PASSED',flush=True)
