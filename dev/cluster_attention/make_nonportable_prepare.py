"""Fuse preparation with 16-block attention; only ranks 0..7 write new KV."""
from pathlib import Path
import runpy
m=runpy.run_path('dev/cluster_attention/prepare_fused.py')
s=m['CUDA'].replace('__cluster_dims__(8,1,1)','__cluster_dims__(16,1,1)')
s=s.replace('  int prep_token=b*8+split,col=(tid&15)*4,head=tid>>4;','  if(split<8){\n  int prep_token=b*8+split,col=(tid&15)*4,head=tid>>4;')
s=s.replace('  prep_cluster.sync();\n  for(int i=tid;', '  }\n  prep_cluster.sync();\n  for(int i=tid;',1)
s=s.replace('int row=split*8+warp*2+(lane>>4),d=(lane&15)*4;', 'if(tid<64){\n  int row=split*4+(tid>>4),d=(lane&15)*4;')
s=s.replace('for(int r=0;r<8;r++)','for(int r=0;r<16;r++)')
s=s.replace('  cluster.sync();\n\n#endif','  }\n  cluster.sync();\n\n#endif')
s=s.replace('<<<dim3(8,2,B)','<<<dim3(16,2,B)').replace('tn,page,2048,8,chunk','tn,page,2048,16,chunk')
needle=' cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);'
s=s.replace(needle,''' cudaError_t opt=cudaFuncSetAttribute(wg5_partials,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 if(opt!=cudaSuccess)return opt;
'''+needle)
stats=s.replace('*inv*value_scale;', '*value_scale;')
stats=stats.replace('if(t==0)shared[4096+row]=ls>0.f?m_run[h]+log2f(ls)-8.f:-INFINITY;','if(t==0){shared[4096+row]=m_run[h];shared[4160+row]=ls;}')
stats=stats.replace('float w=mx==-INFINITY?0.f:exp2f(mx-high);den+=w;','float w=mx==-INFINITY?0.f:exp2f(mx-high);den=fmaf(remote[4160+row],w,den);')
assert s!=stats
Path('dev/cluster_attention/nonportable_prepare.cu').write_text(s)
old=Path('dev/cluster_attention/nonportable.py').read_text()
Path('dev/cluster_attention/nonportable_bundle.py').write_text(old+'\nPREP_CPP='+repr(m['CPP'])+'\nPREP_CUDA='+repr(s)+'\nSTATS_CUDA='+repr(stats)+'\n')
