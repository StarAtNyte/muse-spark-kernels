"""Two warp groups, two KV heads, cooperative full-row K/V loads."""
from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_joint_serial.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
s=s.replace('__launch_bounds__(128, 4)','__launch_bounds__(256, 2)')
s=s.replace('wgi = 0, tw = tid & 127','wgi = tid >> 7, tw = tid & 127')
s=s.replace('kh = blockIdx.y, b = blockIdx.z','kh = wgi, b = blockIdx.y')
s=s.replace('asm volatile("bar.sync %0, 128;\\n" ::"r"(1 + wgi) : "memory");','__syncthreads();')
old='  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);'
new=old+'''
  const int head_stride = (4096 + 4096 + WS * 8192 + ((chunk+255)/256)*256*4 + 1023) & ~1023;
  uint8_t* shared_all=smem;
  smem += (threadIdx.x >> 7)*head_stride;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('for (int i = tid; i < n; i += 128)','for (int i = tw; i < n; i += 128)')
a=s.index('  const uint32_t stages_s =');b=s.index('  int wr_stage =',a)
s=s[:a]+r'''
  // Each warp fetches a complete 128-byte two-head row. The destination
  // separates heads into independent WGMMA shared-memory layouts.
  const int ci=tid&15,load_head=(ci>>2)&1,cc=ci&3;
  const bool isv=ci>=8;
  const uint8_t* srcb=(isv?VC:KC)+load_head*64+cc*16;
  uint8_t* dst_stage=shared_all+load_head*head_stride+8192;
  auto issue = [&](int tile,int stage){
    if(tile<nst && !(dbg&8)){
#pragma unroll
      for(int i=0;i<4;i++){
        int key=(tid>>4)+16*i,li=tile*WT+key;
        long rr=li<n?rows[li]:0;
        int off=isv?4096+key*64+(cc^(key&3))*16:core_off(key,cc);
        cp16(dst_stage+stage*8192+off,srcb+rr*128);
      }
    }
    cp_commit();
  };
''' +s[b:]
s=s.replace('const size_t dyn = 1024 + 4096 + 4096 + (size_t)WS * 8192 + rmax * 4;', 'const size_t dyn = 1024 + 2*((4096 + 4096 + (size_t)WS * 8192 + rmax * 4 + 1023)&~(size_t)1023);')
s=s.replace('dim3 grid(splits, 2, B);','dim3 grid(splits, B);').replace('<<<grid, 128, dyn, st>>>','<<<grid, 256, dyn, st>>>')
lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WG_CUDA = '+repr(s)+'\n']
out=''.join(lines).replace('muse_wg5_fp8p_joint_serial','muse_wg5_fp8p_paired_heads')
ast.parse(out)
Path('dev/fp8_probability/paired_heads.cu').write_text(s)
Path('dev/ptx_campaign/attention_fp8p_paired_heads.py').write_text(out)
