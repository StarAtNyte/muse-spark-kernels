from pathlib import Path
import ast
v={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(Path('dev/moe_xpack/probe.py').read_text()).body if isinstance(n,ast.Assign)}
s=v['CUDA'];a=s.index('template <int MODE, int NTB, int NCH, int S, int DBG>');b=s.index('template <int MODE, int NTB, int NMAX, int S, int DBG>',a)
body=s[a:b]
old='''          OUT[(size_t)p * ID + (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row] = __float2bfloat16(sil * up);'''
new='''          const int q = (bx * (NTC / 2) + ((warp * NTB + n) >> 1)) * 16 + row;
          const size_t oi = (DBG & 256) ? (((size_t)blockIdx.y * KC + q / 64) * 256 + ((col / 8 * 2 + (q % 16) / 8) * 32 + col % 8 * 4 + (q % 64) / 16)) * 8 + q % 8 : (size_t)p * ID + q;
          OUT[oi] = __float2bfloat16(sil * up);'''
assert old in body;body=body.replace(old,new);s=s[:a]+body+s[b:]
# Guard invalid packed columns rather than consuming unwritten padding.
s=s.replace('    xok[r] = true;','    xok[r] = (sl >> 6) * 8 + ((sl & 31) >> 2) < ntok;')
a=s.index('extern "C" int wm_gemm_c(');b=s.index('#include <torch/extension.h>',a);section=s[a:b]
line=next(l for l in section.splitlines() if 'if(mode==0' in l and 'dbg==0)' in l)
section=section.replace('return 1;',line.replace('dbg==0','dbg==256').replace('<0,4,4,4,0>','<0,4,4,4,256>')+'\nreturn 1;')
s=s[:a]+section+s[b:];v['CUDA']=s
Path('dev/moe_xfused').mkdir(exist_ok=True);Path('dev/moe_xfused/probe.py').write_text('\n'.join(k+'='+repr(x) for k,x in v.items())+'\n');Path('dev/moe_xfused/kernel.cu').write_text(s)
base=Path('dev/modal/drv_moe_kpair.py').read_text().split('def capture(data,dbg):')[0].replace('moe_kpair_v1','moe_xfused_v1')
base+='''
n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
src=ast.get_source_segment(m.PARENT,n).replace('def kernel_fn(','def fused_forward(')
src=src.replace('    inter = torch.empty((m * TOP_K, 4096), dtype=torch.bfloat16, device=dev)','    inter = torch.empty((maxslots,64,256,8) if state["fused"] else (m * TOP_K,4096),dtype=torch.bfloat16,device=dev)')
src=src.replace('nw, ring, _WM_DBG, wm["w13"]','nw, ring, 256 if state["fused"] else 0, wm["w13"]').replace('nw, ring, _WM_DBG, wm["w2"]','nw, ring, 128 if state["fused"] else 0, wm["w2"]')
exec(src,sub.__dict__)
def capture(data,fused):
 state['fused']=fused;sub.fused_forward(data,state);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=sub.fused_forward(data,state)
 g.replay();torch.cuda.synchronize();return g,out
for bs in (128,256):
 for seed in (419,7919,15241,49157):
    data=R.generate_input(bs,'ep8',seed);gb,ob=capture(data,False);g,o=capture(data,True)
    ok,msg=R.check_implementation(data,o,weights);equal=torch.equal(o,ob)
    print('CHECK',bs,seed,ok,msg,'bit_equal',equal,flush=True);assert ok and equal,msg
    bt=[];ct=[]
    for rep in range(24):
        if rep%2:c,b=elapsed(g),elapsed(gb)
        else:b,c=elapsed(gb),elapsed(g)
        bt.append(b);ct.append(c)
    print('RESULT',json.dumps(dict(batch=bs,seed=seed,baseline_us=statistics.median(bt),candidate_us=statistics.median(ct),ratio=statistics.median(bt)/statistics.median(ct))),flush=True)
print('XFUSED_SCREEN_PASSED',flush=True)
'''
Path('dev/modal/drv_moe_xfused.py').write_text(base)
