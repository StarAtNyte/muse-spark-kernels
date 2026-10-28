"""Pipeline next QK across current V conversion and softmax; preserve async operands."""
import ast
from pathlib import Path
s=Path('experiments/attention_transpose_selected/submission.py').read_text();tree=ast.parse(s)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));cuda=ast.literal_eval(node.value)
cuda=cuda.replace('__launch_bounds__(128, 4)','__launch_bounds__(128, 2)').replace('#define WS 2','#define WS 3')
old='''  int rd_stage = 0;
#pragma unroll 1
'''
new='''  int rd_stage = 0;
  float sacc[32],snext[32];
  uint32_t pa[8][2];
  if(nst>0){
    cp_wait<WS-2>();proxy_fence();__syncthreads();
    wg_fence();wg_qk_first(sacc,qd0,kd0);wg_qk_acc(sacc,qd1,kd0+16);wg_commit_wait();
  }
#pragma unroll 1
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
old='''    cp_wait<WS - 2>();
    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }
'''
hold='asm volatile("" : '+', '.join(f'"+r"(pa[{i}][{j}])' for i in range(8) for j in range(2))+' ::"memory");'
new='''    if(s>0){wg_wait0();'''+hold+'''
#pragma unroll
      for(int i=0;i<36;i++)asm volatile("":"+f"(o[i])::"memory");
    }
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
old='''    float sacc[32];
    wg_fence();
    wg_qk_first(sacc, qd0, kdesc);
    wg_qk_acc(sacc, qd1, kdesc + 16);
    wg_commit();
'''
new='''    if(s+1<nst){
      cp_wait<WS-2>();proxy_fence();__syncthreads();
      const uint64_t next_k=kd0+(uint64_t)(rd_stage*512);
      wg_fence();wg_qk_first(snext,qd0,next_k);wg_qk_acc(snext,qd1,next_k+16);wg_commit();
    }
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
# Current scores were completed at the previous iteration's wait (or prologue).
assert cuda.count('    wg_wait0();')==1;cuda=cuda.replace('    wg_wait0();','')
assert cuda.count('    uint32_t pa[8][2];')==1;cuda=cuda.replace('    uint32_t pa[8][2];','')
old='''    wg_commit_wait();
#pragma unroll
    for (int i = 0; i < 36; i++) asm volatile("" : "+f"(o[i])::"memory");
'''
new='''    wg_commit();
    if(s+1<nst){
      asm volatile("wgmma.wait_group.sync.aligned 1;\\n" ::: "memory");
#pragma unroll
      for(int i=0;i<32;i++)asm volatile("":"+f"(snext[i])::"memory");
#pragma unroll
      for(int i=0;i<32;i++)sacc[i]=snext[i];
    }else{wg_wait0();'''+hold+'''
#pragma unroll
      for(int i=0;i<36;i++)asm volatile("":"+f"(o[i])::"memory");
    }
'''
assert cuda.count(old)==1;cuda=cuda.replace(old,new)
lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(cuda)+'\n'];s=''.join(lines).replace('load_inline("muse_wg5_onehead_s2",','load_inline("muse_wg5_overlap_softmax_v2",')
ast.parse(s);Path('dev/ptx_campaign/attention_verify_overlap_v2.py').write_text(s)
Path('dev/verify_overlap/kernel_v2.cu').write_text(cuda)
