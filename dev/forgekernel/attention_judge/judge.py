"""Independent page-1 sliding attention math with explicit FP8 cache updates."""
import torch


def _reference(qkv, seq_lens, cache_loc, k_cache, v_cache, page_table, k_scale, v_scale, query_fp8=False, new_kv=None):
    batch, length = qkv.shape[:2]
    qi, ki, vi, gate = qkv.split((1024,128,128,1024), dim=-1)
    q = qi.reshape(batch,length,16,64).float()
    k = ki.reshape(batch,length,2,64).float()
    v = vi.reshape(batch,length,2,64).float()
    q = q * torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-5) * 5.4730064863875
    k = k * torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-5)
    pos = seq_lens.long()[:,None] + torch.arange(length,device=q.device)
    freq = 500000.0 ** (-torch.arange(0,64,2,device=q.device).float()/64)
    ang = pos.float()[...,None]*freq
    c,s = ang.cos().unsqueeze(-2),ang.sin().unsqueeze(-2)
    def rotate(x):
        even,odd=x[...,::2],x[...,1::2]
        return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)
    q = rotate(q)
    q = (q / 5.4730064863875).to(torch.float8_e4m3fn).float() * 5.4730064863875 if query_fp8 else q.bfloat16().float()
    k = rotate(k)
    kn = (k/k_scale).to(torch.float8_e4m3fn)
    vn = (v/v_scale).to(torch.float8_e4m3fn)
    if new_kv is not None:
        kn, vn = new_kv
    k_cache.view(-1,2,64)[cache_loc.long()] = kn.reshape(-1,2,64)
    v_cache.view(-1,2,64)[cache_loc.long()] = vn.reshape(-1,2,64)
    result = torch.empty((batch,length,16,64),device=q.device)
    for b in range(batch):
        seq=int(seq_lens[b])
        for j in range(length):
            end=seq+j+1
            start=max(0,end-2048)
            rows=page_table[b,start:end].long()
            keys=k_cache.view(-1,2,64)[rows].float()*k_scale
            vals=v_cache.view(-1,2,64)[rows].float()*v_scale
            score=torch.einsum('kgd,tkd->kgt',q[b,j].reshape(2,8,64),keys)*0.125
            result[b,j]=torch.einsum('kgt,tkd->kgd',score.softmax(-1),vals).reshape(16,64)
    result=result*torch.rsqrt(result.square().mean(-1,keepdim=True)+1e-5)
    result=result*torch.sigmoid(gate.float()).reshape(batch,length,16,64)
    return result.reshape(batch,length,1024).bfloat16(),k_cache,v_cache


def run(qkv, seq_lens, cache_loc, k_cache, v_cache, page_table, k_scale, v_scale):
    return _reference(qkv, seq_lens, cache_loc, k_cache, v_cache, page_table, k_scale, v_scale)


def challenge_check(inputs, outputs):
    """Challenge-relative-L2 and new-cache checks; inputs must be pristine clones."""
    if len(inputs)!=8 or len(outputs)!=3:
        return False, 'wrong input or output count'
    qkv, seq, loc, kc, vc, table, ks, vs = inputs
    out, kout, vout = outputs
    if not all(isinstance(x,torch.Tensor) for x in outputs):
        return False, 'outputs must be tensors'
    if out.shape != (*qkv.shape[:2],1024) or out.dtype != torch.bfloat16:
        return False, 'output shape or dtype mismatch'
    if kout.shape!=kc.shape or vout.shape!=vc.shape or kout.dtype!=kc.dtype or vout.dtype!=vc.dtype:
        return False, 'cache shape or dtype mismatch'
    slots=loc.long(); shape=(*qkv.shape[:2],2,64)
    gotk=kout.view(-1,2,64)[slots].reshape(shape); gotv=vout.view(-1,2,64)[slots].reshape(shape)
    def relative(x,y):
        x=x.float(); y=y.float()
        return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp(min=1e-12))
    def compare(x,y,tol):
        if x.shape!=y.shape or x.dtype!=y.dtype or not bool(torch.isfinite(x.float()).all()): return False
        x=x.float(); y=y.float(); diff=x-y
        if relative(x,y)>tol: return False
        d=diff.reshape(-1,y.shape[-1]).norm(dim=-1); r=y.reshape(-1,y.shape[-1]).norm(dim=-1)
        floor=(0.1*r.mean()).clamp(min=1e-12)
        return bool((d/torch.maximum(r,floor)<=5.0*tol).all())
    expected,exk,exv=_reference(qkv,seq,loc,kc.clone(),vc.clone(),table,ks,vs)
    exk=exk.view(-1,2,64)[slots].reshape(shape); exv=exv.view(-1,2,64)[slots].reshape(shape)
    if not compare(gotk.float()*ks,exk.float()*ks,0.08): return False, 'new key cache differs'
    if not compare(gotv.float()*vs,exv.float()*vs,0.08): return False, 'new value cache differs'
    ref,_,_=_reference(qkv,seq,loc,kc.clone(),vc.clone(),table,ks,vs,new_kv=(gotk,gotv))
    fp8,_,_=_reference(qkv,seq,loc,kc.clone(),vc.clone(),table,ks,vs,query_fp8=True,new_kv=(gotk,gotv))
    ref=ref.bfloat16(); fp8=fp8.bfloat16()
    tol=min(0.15,max(0.03,2.5*relative(fp8,ref)+0.015))
    if not compare(out,ref,tol): return False, f'attention output L2 or row check failed (bound {tol:.6g})'
    return True, f'challenge L2 and cache checks passed (bound {tol:.6g})'


def make_inputs(axes, group, seed, device):
    torch.manual_seed(seed)
    b,l,p,t=(axes[k] for k in ('batch','length','pages','table_len'))
    scale=0.125 if group == 'scaled' else 1.0
    qkv=(torch.randn((b,l,2304),device=device)* (1.5 if group == 'scaled' else 1)).bfloat16()
    base=64 if group == 'short' else 4000
    seq=(torch.arange(b,device=device,dtype=torch.int32)*13+base).clamp(max=t-l)
    table=torch.arange(b*t,device=device,dtype=torch.int32).reshape(b,t)
    loc=table.gather(1,seq.long()[:,None]+torch.arange(l,device=device)).reshape(-1).long()
    shape=(p,1,2,64)
    kc=(torch.randn(shape,device=device,dtype=torch.float16)*0.3).to(torch.float8_e4m3fn)
    vc=(torch.randn(shape,device=device,dtype=torch.float16)*0.3).to(torch.float8_e4m3fn)
    ks=torch.tensor(scale,device=device); vs=torch.tensor(scale,device=device)
    return dict(qkv=qkv,seq_lens=seq,cache_loc=loc,k_cache=kc,v_cache=vc,page_table=table,k_scale=ks,v_scale=vs)
