"""Independent page-1 sliding attention math with explicit FP8 cache updates."""
import torch


def run(qkv, seq_lens, cache_loc, k_cache, v_cache, page_table, k_scale, v_scale):
    batch, length = qkv.shape[:2]
    qi, ki, vi, gate = qkv.split((1024,128,128,1024), dim=-1)
    q = qi.reshape(batch,length,16,64).float()
    k = ki.reshape(batch,length,2,64).float()
    v = vi.reshape(batch,length,2,64).float()
    q = q * torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-5)
    k = k * torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-5)
    pos = seq_lens.long()[:,None] + torch.arange(length,device=q.device)
    freq = 500000.0 ** (-torch.arange(0,64,2,device=q.device).float()/64)
    ang = pos.float()[...,None]*freq
    c,s = ang.cos().unsqueeze(-2),ang.sin().unsqueeze(-2)
    def rotate(x):
        even,odd=x[...,::2],x[...,1::2]
        return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)
    q = rotate(q).to(torch.float8_e4m3fn).float()*5.4730064863875
    k = rotate(k)
    kn = (k/k_scale).to(torch.float8_e4m3fn)
    vn = (v/v_scale).to(torch.float8_e4m3fn)
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
