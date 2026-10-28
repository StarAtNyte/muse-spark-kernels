"""Test narrow packed-head tiles with capped and relaxed register budgets."""
exec(open('/w/modal/drv_attention_merge.py').read().split('\nfor batch in (3, 16, 32, 33):')[0])
from torch.utils.cpp_extension import load_inline
attrs=r'''
std::vector<int64_t> sparse_attrs() {
 cudaFuncAttributes a;auto st=cudaFuncGetAttributes(&a,sparse_decode<2>);
 TORCH_CHECK(st==cudaSuccess,cudaGetErrorString(st));
 return {a.numRegs,(int64_t)a.localSizeBytes};
}
'''
variants={'lazy_loose':candidate._SP_CUDA,
          'lazy_capped':candidate._SP_CUDA.replace('__launch_bounds__(128,1)','__launch_bounds__(128,3)')}
modules={}
for name,src in variants.items():
    modules[name]=load_inline('muse_sparse_lazy_'+name,candidate._SP_CPP+'\nstd::vector<int64_t> sparse_attrs();',
        cuda_sources=src+attrs,functions=['sparse_decode_run','sparse_attrs'],
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
    print('RESOURCE',name,list(modules[name].sparse_attrs()),flush=True)
candidate._SP_MOD=modules['lazy_loose']
data=R.generate_input(1,3000,'tp8','full',431,page=64)
out=candidate.kernel_fn(data)
ok,message=R.check_implementation(data,R.collect(data,out))
print('EARLY_CHECK',ok,message,flush=True)
if not ok:raise AssertionError(message)
sys.path.insert(0,'/w/modal')
from compile_split import build
import time
compile_start=time.time()
base._FD_MOD=build('muse_fd_attn_split',base._FD_CPP,base._FD_CUDA,
    ['fd_run_t','slide_run_t','pf_t','fd2_set_dbg_t','fd_set_pdl_t'],['-O3']+base._FD_FLAGS)
base._FD_MOD.fd_set_pdl_t(1)
print('BASE_COMPILE_SECONDS',time.time()-compile_start,flush=True)
for layer in ('sliding','full'):
    for batch in (1,16,32):
        options=((16,32) if batch==1 else (8,16) if batch==16 else (4,8)) if layer=='sliding' else ((128,256) if batch==1 else (64,128) if batch==16 else (32,64))
        for seed in (539,1427):
            data=R.generate_input(batch,80000,'tp8',layer,seed,page=1,spread='agentic',
                dist='soft' if layer=='sliding' else 'coherent')
            gb,ob=capture(base,data)
            for name,mod in modules.items():
                candidate._SP_MOD=mod
                for splits in options:
                    candidate._SP_SPLITS_OVERRIDE=splits
                    gc,oc=capture(candidate,data)
                    err=((ob.float()-oc.float()).norm()/ob.float().norm()).item()
                    if not torch.isfinite(oc).all().item() or err>.003:raise AssertionError((layer,batch,seed,name,err))
                    bt,ct,ratios=[],[],[]
                    for rep in range(16):
                        if rep%2:c,b=elapsed(gc),elapsed(gb)
                        else:b,c=elapsed(gb),elapsed(gc)
                        bt.append(b);ct.append(c);ratios.append(b/c)
                    print('RESULT',json.dumps(dict(variant=name,layer=layer,batch=batch,seed=seed,splits=splits,
                        error=err,speedup=statistics.median(ratios),baseline_us=statistics.median(bt),
                        candidate_us=statistics.median(ct))),flush=True)
                    del gc,oc
            del gb,ob,data
