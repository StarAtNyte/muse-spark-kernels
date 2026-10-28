"""Keep one WGMMA group in flight while decoding into the other A buffer."""
from pathlib import Path
import runpy
m=runpy.run_path('dev/moe_wgnarrow/probe.py');s=m['CUDA']
a=s.index('__device__ __forceinline__ void wm_body_wg(');b=s.index('template <int MODE, int NMAX, int S>',a)
body=s[a:b]
body=body.replace('for (int c = 0; c < S - 1; c++) issue(c);','for (int c = 0; c < S - 2; c++) issue(c);')
needle='''#pragma unroll 1
  for (int i = 0; i < KC; i++) {'''
buffers='''  uint32_t a0buf[NG][4][4]={},a1buf[NG][4][4]={};
'''
assert body.count(needle)==1;body=body.replace(needle,buffers+needle)
wait='''
#pragma unroll
    for(int q=0;q<NG;q++)for(int k=0;k<4;k++)for(int r=0;r<4;r++)
      asm volatile("" : "+r"(a0buf[q][k][r]), "+r"(a1buf[q][k][r])::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
    for(int q=0;q<NG;q++)for(int k=0;k<4;k++)for(int r=0;r<4;r++)
      asm volatile("" : "+r"(a0buf[q][k][r]), "+r"(a1buf[q][k][r])::"memory");
'''
body=body.replace('    cp_wait<S - 2>();',wait+'    cp_wait<S - 3>();')
body=body.replace('    issue(i + S - 1);','    issue(i + S - 2);')
body=body.replace('    uint32_t a[NG][4][4];\n','')
a0=body.index('#pragma unroll\n    for (int q = 0; q < NG; q++) {',body.index('const int cq = i & 3;'))
b0=body.index('    wg_commit_wait();',a0)
compute=body[a0:b0]
body=body[:a0]+'''    if(i&1){
'''+compute.replace('a[','a1buf[')+'''    }else{
'''+compute.replace('a[','a0buf[')+'''    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#pragma unroll
    for(int q=0;q<NG;q++)for(int k=0;k<4;k++)for(int r=0;r<4;r++)
      asm volatile("" : "+r"(a0buf[q][k][r]), "+r"(a1buf[q][k][r])::"memory");
'''+body[b0+len('    wg_commit_wait();\n'):]
body=body.replace('  cp_wait<0>();','  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");\n  cp_wait<0>();')
s=s[:a]+body+s[b:]
Path('dev/moe_wgnarrow/pipeline.cu').write_text(s)
Path('dev/moe_wgnarrow/pipeline_probe.py').write_text(Path('dev/moe_wgnarrow/probe.py').read_text()+'\nPIPELINE='+repr(s)+'\n')
