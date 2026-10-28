from pathlib import Path
import runpy
m=runpy.run_path('dev/cluster_attention/fused.py')
out='CPP='+repr(m['CPP'])+'\n'
for cs in (1,2,4,8):
 s=m['CUDA'].replace('__cluster_dims__(8,1,1)',f'__cluster_dims__({cs},1,1)')
 s=s.replace('int row=split*8+warp*2+(lane>>4),d=(lane&15)*4;',f'for(int rr=0;rr<{8//cs};rr++){{\n  int row=split*{64//cs}+rr*8+warp*2+(lane>>4),d=(lane&15)*4;')
 s=s.replace('for(int r=0;r<8;r++)',f'for(int r=0;r<{cs};r++)')
 s=s.replace('  cluster.sync();\n\n#endif','  }\n  cluster.sync();\n\n#endif')
 assert f'for(int rr=0;rr<{8//cs};rr++)' in s
 s=s.replace('<<<dim3(8,2,B)',f'<<<dim3({cs},2,B)').replace('HQ,tn,page,window,8,chunk','HQ,tn,page,window,'+str(cs)+',chunk')
 Path(f'dev/cluster_attention/fused_cs{cs}.cu').write_text(s)
 out+=f'CUDA_CS{cs}='+repr(s)+'\n'
Path('dev/cluster_attention/sizes.py').write_text(out)
