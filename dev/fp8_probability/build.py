"""Match the pinned baseline's FP8 P/V multiply and FP32 softmax sums."""
from pathlib import Path
import ast
s=Path('experiments/attention_transpose_selected/submission.py').read_text()
s+=r'''

# The pinned sgl-attn mainloop converts P to its FP8 Element type and uses
# Softmax<..., Max_offset=8>. Keep sums in FP32, as that baseline does.
@triton.jit
def _fp8p_parts(Q,KC,VC,TABLE,SEQ,KS,VS,PART,LSE,
 HQ:tl.constexpr,HKV:tl.constexpr,L:tl.constexpr,PAGE:tl.constexpr,TN:tl.constexpr,
 WINDOW:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,BN:tl.constexpr,
 SM:tl.constexpr,SCALED:tl.constexpr):
 b,kh,split=tl.program_id(0),tl.program_id(1),tl.program_id(2)
 r=tl.arange(0,64);d=tl.arange(0,64);token=r//8;qh=kh*8+r%8
 prefix=tl.load(SEQ+b);first=tl.maximum(0,prefix-WINDOW+1) if WINDOW else 0
 begin=first+split*CHUNK;end=tl.minimum(begin+CHUNK,prefix+L)
 q=tl.load(Q+((b*L+token[:,None])*HQ+qh[:,None])*64+d[None,:],token[:,None]<L,0).to(tl.float8e4nv,bitcast=True)
 ks=tl.load(KS) if SCALED else 1.;vs=tl.load(VS) if SCALED else 1.
 m=tl.full((64,),-float('inf'),tl.float32);total=tl.full((64,),0,tl.float32)
 out=tl.full((64,64),0,tl.float32)
 for start in range(begin,end,BN):
  key=start+tl.arange(0,BN);pg=key//PAGE
  phys=tl.load(TABLE+b*TN+pg,(key<end)&(pg<TN),-1)
  rr=tl.maximum(phys,0)*PAGE+key%PAGE
  valid=(key<end)&(phys>=0)
  k=tl.load(KC+(rr[None,:]*HKV+kh)*64+d[:,None],valid[None,:],0).to(tl.float8e4nv,bitcast=True)
  score=tl.dot(q,k).to(tl.float32)*(SM*ks*1.4426950408889634)
  visible=valid[None,:]&(key[None,:]<=prefix+token[:,None])&(token[:,None]<L)
  if WINDOW:visible=visible&(key[None,:]>prefix+token[:,None]-WINDOW)
  score=tl.where(visible,score,-float('inf'))
  new=tl.maximum(m,tl.max(score,1));safe=tl.where(new==-float('inf'),0.,new)
  correction=tl.exp2(m-safe)
  p=tl.exp2(score-safe[:,None]+8.)
  total=total*correction+tl.sum(p,1);out*=correction[:,None]
  v=tl.load(VC+(rr[:,None]*HKV+kh)*64+d[None,:],valid[:,None],0).to(tl.float8e4nv,bitcast=True)
  out=tl.dot(p.to(tl.float8e4nv),v,out)
  m=new
 inv=1./tl.where(total>0,total,1.)
 base=((b*HKV+kh)*SPLITS+split)*64+r
 tl.store(PART+base[:,None]*64+d[None,:],out*inv[:,None]*vs)
 logs=tl.where(total>0,(m+tl.log2(total)-8.)*0.6931471805599453,-float('inf'))
 tl.store(LSE+base,logs)

_FP8P_BN=64
_FP8P_SPLITS=None

def _fp8p_forward(data,q,qkv):
 cfg=data['config'];batch=cfg['batch'];length=cfg['q_len'];window=cfg['window']
 bn=_FP8P_BN
 splits=_FP8P_SPLITS or max(4,triton.next_power_of_2(triton.cdiv(256 if window else 1024,batch*2)))
 span=window+length-1 if window else data['page_table'].shape[1]*cfg['page_size']
 chunk=triton.cdiv(triton.cdiv(span,splits),bn)*bn
 part=torch.empty((batch,2,splits,64,64),dtype=torch.float32,device=qkv.device)
 lse=torch.empty((batch,2,splits,64),dtype=torch.float32,device=qkv.device)
 out=torch.empty((batch,length,1024),dtype=torch.bfloat16,device=qkv.device)
 _fp8p_parts[(batch,2,splits)](q.view(torch.uint8),data['k_cache'].view(torch.uint8),data['v_cache'].view(torch.uint8),data['page_table'],data['seq_lens'],
  data['k_scale'],data['v_scale'],part,lse,16,2,length,cfg['page_size'],data['page_table'].shape[1],
  window,splits,chunk,bn,cfg['sm_scale']*cfg['qk_scale_factor'],int(cfg['kv_scale']!=1.),num_warps=4,num_stages=2)
 _merge_norm_gate[(batch*length,16)](part,lse,qkv,out,16,2,64,length,qkv.stride(0),splits,64,num_warps=4)
 return out
'''
needle="    if (cfg['q_len'] == 1 and 8 <= cfg['batch'] <= 64"
new="""    if (cfg['q_len'] == 8 and cfg['batch'] <= 64 and (cfg['window'] or cfg['batch'] >= 8)
            and cfg['num_q_heads'] == 16 and cfg['num_kv_heads'] == 2):
        qkv,q=_fd_prepare(data)
        return _fp8p_forward(data,q,qkv)
"""+needle
assert s.count(needle)==1;s=s.replace(needle,new)
ast.parse(s);Path('dev/ptx_campaign/attention_fp8p_triton.py').write_text(s)
