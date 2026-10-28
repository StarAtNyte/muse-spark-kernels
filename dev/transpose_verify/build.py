"""Verify all eight tokens with 16 warps and one shared FP16 V conversion."""
from pathlib import Path
import ast
s=Path('dev/transpose_head/decode.cu').read_text()
s=s.replace('template<int ST>','__device__ __forceinline__ int voff(int key,int byte){return key*256+(((byte>>4)^(key&15))<<4)+(byte&15);}\ntemplate<int ST>')
s=s.replace('__launch_bounds__(128,2) trans_decode','__launch_bounds__(512,2) trans_verify')
s=s.replace('part=warp>>1','qt=warp>>1').replace('prefix+1','prefix+8').replace('(n+31)/32','(n+15)/16')
s=s.replace('mem+ST*8192','mem+ST*4096+4096').replace('i<n;i+=128','i<n;i+=512')
s=s.replace('Q+(long)b*1024','Q+((long)b*8+qt)*1024')
s=s.replace('(tile%ST)*8192','(tile%ST)*4096').replace('i<256;i+=128','i<128;i+=512').replace('li=tile*32+key','li=tile*16+key').replace('dst+4096','dst+2048')
s=s.replace('part*16+(lane&7)','(lane&7)')
s=s.replace('s[r]=(tile*32+part*16+g+(r>=2?8:0)<n)?s[r]*scale:-INFINITY;', 's[r]=(tile*16+g+(r>=2?8:0)<n && begin+tile*16+g+(r>=2?8:0)<=prefix+qt)?s[r]*scale:-INFINITY;')
a=s.index('  float mx[2]=')
s=s[:a]+'''  uint8_t* vf=mem+ST*4096;
  if(tid<128){int key=tid>>3,byte=(tid&7)*16;uint4 u=*reinterpret_cast<const uint4*>(raw+2048+off(key,byte));
   uint4 a=make_uint4(cv2(u.x),cv2(u.x>>16),cv2(u.y),cv2(u.y>>16));
   uint4 b4=make_uint4(cv2(u.z),cv2(u.z>>16),cv2(u.w),cv2(u.w>>16));
   *reinterpret_cast<uint4*>(vf+voff(key,byte*2))=a;*reinterpret_cast<uint4*>(vf+voff(key,byte*2+16))=b4;
  }
'''+s[a:]
a=s.index('#pragma unroll\n  for(int pr=0;pr<2;pr++)');z=s.index('  __syncthreads();\n }',a)
s=s[:a]+'''  __syncthreads();
#pragma unroll
  for(int pr=0;pr<4;pr++){uint32_t r[4];ldt(r,vf+voff((lane&7)+8*((lane>>3)&1),h*128+pr*32+16*(lane>>4)));
   uint32_t a[4]={r[0],r[2],r[1],r[3]};pmma(o[pr],a,z);
  }
'''+s[z:]
a=s.index(' float* smm=');z=s.index('\nvoid sparse_decode_run',a)
s=s[:a]+'''
#pragma unroll
 for(int c=0;c<2;c++){
  int row=qt*8+2*t+c;long base=((long)(b*2+h)*splits+split)*64+row;
  float inv=ll[c]>0.f?1.f/ll[c]:0.f;
  if(g==0)LSE[base]=ll[c]>0.f?(mm[c]+log2f(ll[c]))*0.6931471805599453f:-INFINITY;
#pragma unroll
  for(int pr=0;pr<4;pr++){int d=pr*16+g;PART[base*64+d]=o[pr][c]*inv*vscale;PART[base*64+d+8]=o[pr][c+2]*inv*vscale;}
 }
}
'''+s[z:]
s=s.replace('sparse_decode_run','trans_verify_run').replace('2*8192+','3*4096+4096+').replace('trans_decode<2>','trans_verify<3>').replace('seq.numel()),128,dyn','seq.numel()),512,dyn')
assert 'part*' not in s and 'trans_decode' not in s
Path('dev/transpose_verify/kernel.cu').write_text(s)
parent=Path('experiments/attention_transpose_selected/submission.py').read_text()
cpp='void trans_verify_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,int64_t);'
parent+='\n_TV_CPP='+repr(cpp)+'\n_TV_CUDA='+repr(s)+'\n'
parent+='''
_TV_MOD=None
_TV_SPLITS=None

def _tv_module():
 global _TV_MOD
 if _TV_MOD is None:
  from torch.utils.cpp_extension import load_inline
  _TV_MOD=load_inline('muse_trans_verify_v1',_TV_CPP,cuda_sources=_TV_CUDA,functions=['trans_verify_run'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return _TV_MOD

def _tv_forward(data,q,qkv):
 cfg=data['config'];batch=cfg['batch']
 span=data['page_table'].shape[1]*cfg['page_size']
 splits=_TV_SPLITS or max(8,triton.next_power_of_2(triton.cdiv(512,batch)))
 splits=max(splits,triton.cdiv(span,8192))
 chunk=triton.cdiv(triton.cdiv(span,splits),16)*16
 partial=torch.empty((batch,2,splits,64,64),dtype=torch.float32,device=qkv.device)
 lse=torch.empty((batch,2,splits,64),dtype=torch.float32,device=qkv.device)
 _tv_module().trans_verify_run(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],data['v_scale'],partial,lse,cfg['page_size'],0,splits,chunk,cfg['sm_scale']*cfg['qk_scale_factor'],int(cfg['kv_scale']!=1.))
 out=torch.empty((batch,8,1024),dtype=torch.bfloat16,device=qkv.device)
 _merge_norm_gate[(batch*8,16)](partial,lse,qkv,out,16,2,64,8,qkv.stride(0),splits,64,num_warps=4)
 return out
'''
needle="    if (cfg['q_len'] == 1 and 8 <= cfg['batch'] <= 64"
new="""    if (cfg['q_len'] == 8 and not cfg['window'] and 8 <= cfg['batch'] <= 64
            and cfg['num_q_heads'] == 16 and cfg['num_kv_heads'] == 2
            and ctx_bound <= (1<<20)+64 and cfg['page_size'] & (cfg['page_size']-1) == 0):
        qkv,q=_fd_prepare(data)
        return _tv_forward(data,q,qkv)
"""+needle
assert parent.count(needle)==1;parent=parent.replace(needle,new)
ast.parse(parent);Path('dev/ptx_campaign/attention_trans_verify_v1.py').write_text(parent)
