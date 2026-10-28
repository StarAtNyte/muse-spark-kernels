from pathlib import Path
import runpy
old=runpy.run_path('dev/cluster_attention/fused.py');new=runpy.run_path('dev/cluster_attention/prepare_fused.py')
s=new['CUDA'].replace('*inv*value_scale;', '*value_scale;')
before='if(t==0)shared[4096+row]=ls>0.f?m_run[h]+log2f(ls)-8.f:-INFINITY;'
after='if(t==0){shared[4096+row]=m_run[h];shared[4160+row]=ls;}'
assert before in s;s=s.replace(before,after)
before='float w=mx==-INFINITY?0.f:exp2f(mx-high);den+=w;'
after='float w=mx==-INFINITY?0.f:exp2f(mx-high);den=fmaf(remote[4160+row],w,den);'
assert before in s;s=s.replace(before,after)
Path('dev/cluster_attention/prepare_stats.cu').write_text(s)
Path('dev/cluster_attention/prepare_bundle.py').write_text('_OLD_CPP='+repr(old['CPP'])+'\n_OLD_CUDA='+repr(old['CUDA'])+'\nCPP='+repr(new['CPP'])+'\n_PREP_CUDA='+repr(new['CUDA'])+'\n_STATS_CUDA='+repr(s)+'\n')
