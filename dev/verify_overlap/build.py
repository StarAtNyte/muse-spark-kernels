"""Overlap next-tile QK with current PV, holding async register operands live."""
import ast
from pathlib import Path
s=Path('experiments/attention_transpose_selected/submission.py').read_text();tree=ast.parse(s)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));cuda=ast.literal_eval(node.value)
cuda=cuda.replace('__launch_bounds__(128, 4)','__launch_bounds__(128, 2)')
old='''  int rd_stage = 0;
#pragma unroll 1
'''
new='''  int rd_stage = 0;
  float sacc[32];
  if(nst>0){
    cp_wait<0>();proxy_fence();__syncthreads();
    wg_fence();wg_qk_first(sacc,qd0,kd0);wg_qk_acc(sacc,qd1,kd0+16);wg_commit();
  }
#pragma unroll 1
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
old='''    cp_wait<WS - 2>();
    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,'')
old='''    float sacc[32];
    wg_fence();
    wg_qk_first(sacc, qd0, kdesc);
    wg_qk_acc(sacc, qd1, kdesc + 16);
    wg_commit();
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,'')
old='''    wg_commit_wait();
#pragma unroll
    for (int i = 0; i < 36; i++) asm volatile("" : "+f"(o[i])::"memory");
'''
hold='asm volatile("" : '+', '.join(f'"+r"(pa[{i}][{j}])' for i in range(8) for j in range(2))+' ::"memory");'
new='''    wg_commit();
    if(s+1<nst){
      cp_wait<0>();proxy_fence();__syncthreads();
      const uint64_t next_k=kd0+(uint64_t)(rd_stage*512);
      wg_fence();wg_qk_first(sacc,qd0,next_k);wg_qk_acc(sacc,qd1,next_k+16);wg_commit();
      asm volatile("wgmma.wait_group.sync.aligned 1;\\n" ::: "memory");
    }else{wg_wait0();}
    '''+hold+'''
#pragma unroll
    for (int i = 0; i < 36; i++) asm volatile("" : "+f"(o[i])::"memory");
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(cuda)+'\n'];s=''.join(lines).replace('load_inline("muse_wg5_onehead_s2",','load_inline("muse_wg5_overlap_qkpv_v1",')
ast.parse(s);Path('dev/ptx_campaign/attention_verify_overlap_v1.py').write_text(s)
Path('dev/verify_overlap/kernel.cu').write_text(cuda)
