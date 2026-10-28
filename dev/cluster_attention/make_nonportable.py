"""Sixteen-block H200 cluster: four output rows per rank at the DSM merge."""
from pathlib import Path
import runpy
m=runpy.run_path('dev/cluster_attention/fused.py')
s=m['CUDA'].replace('__cluster_dims__(8,1,1)','__cluster_dims__(16,1,1)')
s=s.replace('int row=split*8+warp*2+(lane>>4),d=(lane&15)*4;',
'''if(tid<64){
  int row=split*4+(tid>>4),d=(lane&15)*4;''')
s=s.replace('for(int r=0;r<8;r++)','for(int r=0;r<16;r++)')
s=s.replace('  cluster.sync();\n\n#endif','  }\n  cluster.sync();\n\n#endif')
s=s.replace('<<<dim3(8,2,B)','<<<dim3(16,2,B)').replace('HQ,tn,page,window,8,chunk','HQ,tn,page,window,16,chunk')
needle=' cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);'
s=s.replace(needle,''' cudaError_t opt=cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 if(opt!=cudaSuccess)return opt;
'''+needle)
marker='#include <torch/extension.h>'
capacity='''
extern "C" int cluster_capacity_c(int chunk,int B){
 int dyn=1024+4096+4096+WS*8192+((chunk+255)/256)*256*4;
 cudaError_t e=cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 if(e!=cudaSuccess)return -int(e);
 e=cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);
 if(e!=cudaSuccess)return -int(e);
 cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16,2,B);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=dyn;
 int active=0;e=cudaOccupancyMaxActiveClusters(&active,wg5_partials,&cfg);
 return e==cudaSuccess?active:-int(e);
}
'''
s=s.replace(marker,capacity+marker)
s+='''
extern "C" int cluster_capacity_c(int,int);
int64_t cluster_capacity(int64_t chunk,int64_t B){return cluster_capacity_c(chunk,B);}
'''
assert 'if(tid<64)' in s and '  }\n  cluster.sync();\n\n#endif' in s
Path('dev/cluster_attention/nonportable.cu').write_text(s)
Path('dev/cluster_attention/nonportable.py').write_text('CPP='+repr(m['CPP']+'\nint64_t cluster_capacity(int64_t,int64_t);')+'\nCUDA='+repr(s)+'\nCPP8='+repr(m['CPP'])+'\nCUDA8='+repr(m['CUDA'])+'\n')
