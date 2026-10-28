from pathlib import Path
import ast
v={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(Path('dev/moe_bottleneck/probe.py').read_text()).body if isinstance(n,ast.Assign)}
s=v['CUDA'];a=s.index('template <int MODE, int NTB, int NCH, int S, int DBG>');b=s.index('template <int MODE, int NTB, int NMAX, int S, int DBG>',a);body=s[a:b].replace('void wm_body_rs(','void wm_body_packed(')
a1=body.index('  const bf16* xb[RPT];');b1=body.index('  uint32_t sink = 0;',a1)
body=body[:a1]+'''  const bf16* xb[RPT];
  bool xok[RPT];
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    int sl = tid + 128 * r;
    xb[r] = X + ((size_t)blockIdx.y * KC * 256 + sl) * 8;
    xok[r] = true;
  }

'''+body[b1:]
body=body.replace('xb[r] + c * 64','xb[r] + c * 2048')
s=s[:b]+body+s[b:];a=s.index('__global__ void __launch_bounds__(128, (S <= 3 ? 4 : 3))');b=s.index('template <int MODE, int NTB, int NMAX, int S, int DBG>\nstatic void launch_rs',a);piece=s[a:b]
for line in piece.splitlines():
 if 'wm_body_rs<' in line:
  pos=line.index('wm_body_rs<');call=line[pos:];piece=piece.replace(line,line[:pos]+'{ if constexpr (DBG & 128) '+call.replace('wm_body_rs<','wm_body_packed<')+' else '+call+' }')
s=s[:a]+piece+s[b:];a=s.index('extern "C" int wm_gemm_c(');b=s.index('#include <torch/extension.h>',a);section='\n'.join(l for l in s[a:b].splitlines() if not ('if(mode==' in l and 'dbg==0)' not in l))
for mode in (0,1):
 line=next(l for l in section.splitlines() if f'if(mode=={mode}' in l)
 section=section.replace('return 1;',line.replace('dbg==0','dbg==128').replace(f'<{mode},4,4,4,0>',f'<{mode},4,4,4,128>')+'\nreturn 1;')
s=s[:a]+section+'\n'+s[b:]
marker=s.index('#include <torch/extension.h>')
packing=r'''
template<int MODE>
__global__ void wm_pack_x(const bf16* X,uint4* P,const int* lst,const int* act,int lmax){
 int slot=blockIdx.y,c=blockIdx.x,sl=threadIdx.x;
 int4 item=__ldcg(reinterpret_cast<const int4*>(act)+slot);
 int h=sl>>6,j=(sl>>5)&1,L=sl&31,col=h*8+(L>>2);
 uint4 v=make_uint4(0,0,0,0);
 if(col<item.z){
   int p=__ldcg(lst+item.x*lmax+item.y+col);
   const bf16* src=X+(size_t)(MODE==0?(p>>3):p)*4096+c*64+16*(L&3)+8*j;
   v=__ldg(reinterpret_cast<const uint4*>(src));
 }
 P[((size_t)slot*64+c)*256+sl]=v;
}
extern "C" void wm_pack_x_c(int mode,const void* X,void* P,const int* lst,const int* act,int maxslots,int lmax,cudaStream_t st){
 if(mode==0)wm_pack_x<0><<<dim3(64,maxslots),256,0,st>>>((const bf16*)X,(uint4*)P,lst,act,lmax);
 else wm_pack_x<1><<<dim3(64,maxslots),256,0,st>>>((const bf16*)X,(uint4*)P,lst,act,lmax);
}
'''
s=s[:marker]+packing+s[marker:]+r'''
extern "C" void wm_pack_x_c(int,const void*,void*,const int*,const int*,int,int,cudaStream_t);
void wm_pack_run(int64_t mode,torch::Tensor X,torch::Tensor P,torch::Tensor lst,torch::Tensor act,int64_t maxslots,int64_t lmax){
 wm_pack_x_c(mode,X.data_ptr(),P.data_ptr(),lst.data_ptr<int>(),act.data_ptr<int>(),maxslots,lmax,at::cuda::getCurrentCUDAStream());
}
'''
v['CUDA']=s;v['CPP']+='\nvoid wm_pack_run(int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
Path('dev/moe_xpack').mkdir(exist_ok=True);Path('dev/moe_xpack/probe.py').write_text('\n'.join(k+'='+repr(x) for k,x in v.items())+'\n');Path('dev/moe_xpack/kernel.cu').write_text(s)
base=Path('dev/modal/drv_moe_kpair.py').read_text().split('def capture(data,dbg):')[0].replace('moe_kpair_v1','moe_xpack_v1').replace("'wm_finish','wm_pf']","'wm_finish','wm_pf','wm_pack_run']")
base+='''
n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
src=ast.get_source_segment(m.PARENT,n).replace('def kernel_fn(','def packed_forward(')
for mode,x,y,wp,sf,gs,nt in [(0,'x','inter','w13','s13','gs13','nt'),(1,'inter','y','w2','s2','gs2','nt1')]:
 old=f'    mod.wm_gemm_run({mode}, nch, {nt}, nw, ring, _WM_DBG, wm["{wp}"], wm["{sf}"], wm["{gs}"], {x}, {y}, cnt, lst, act, nact, woff, maxslots, lmax)'
 new=f'    xp={x}\\n    dbg=0\\n    if state["pack"][{mode}]:\\n        xp=torch.empty((maxslots,64,256,8),dtype=x.dtype,device=dev)\\n        mod.wm_pack_run({mode},{x},xp,lst,act,maxslots,lmax)\\n        dbg=128\\n    mod.wm_gemm_run({mode}, nch, {nt}, nw, ring, dbg, wm["{wp}"], wm["{sf}"], wm["{gs}"], xp, {y}, cnt, lst, act, nact, woff, maxslots, lmax)'
 assert old in src;src=src.replace(old,new)
exec(src,sub.__dict__)
def capture(data,pack):
 state['pack']=pack;sub.packed_forward(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=sub.packed_forward(data,state)
 g.replay();torch.cuda.synchronize();return g,out
for bs in (128,256):
 for seed in (419,7919,15241):
    data=R.generate_input(bs,'ep8',seed);gb,ob=capture(data,(0,0))
    for pack in ((1,1),(1,0),(0,1)):
        g,o=capture(data,pack);ok,msg=R.check_implementation(data,o,weights);equal=torch.equal(o,ob)
        print('CHECK',bs,seed,pack,ok,msg,'bit_equal',equal,flush=True);assert ok and equal,msg
        bt=[];ct=[]
        for rep in range(24):
            if rep%2:c,b=elapsed(g),elapsed(gb)
            else:b,c=elapsed(gb),elapsed(g)
            bt.append(b);ct.append(c)
        print('RESULT',json.dumps(dict(batch=bs,seed=seed,pack=pack,baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),ratio=statistics.median(bt)/statistics.median(ct))),flush=True)
print('XPACK_SCREEN_PASSED',flush=True)
'''
Path('dev/modal/drv_moe_xpack.py').write_text(base)
