"""Merge sufficient statistics directly; preserve attention's BF16 rounding."""
from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_joint_serial.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
s=s.replace(' * inv * value_scale;', ' * value_scale;')
old='if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls) - 8.f) * 0.6931471805599453f : -INFINITY;'
new='if (t == 0) { LSE[base_row*2 + row] = m_run[h]; LSE[base_row*2 + 64 + row] = ls; }'
assert s.count(old)==1;s=s.replace(old,new)
lines=p.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['_WG_CUDA = '+repr(s)+'\n'];p=''.join(lines)
a=p.index('def _wg_forward(');b=p.index('\n_FD_CUDA =',a)
f=p[a:b].replace('splits, 64), device=qkv.device','splits, 128), device=qkv.device')
f=f.replace('_merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)', '_stats_merge[(batch*hkv, triton.cdiv(length*(hq//hkv), _STATS_ROWS))](partial, lse, qkv, result, hq, hkv, length, qkv.stride(0), splits, _STATS_ROWS, num_warps=4)')
assert '_stats_merge[' in f;p=p[:a]+f+p[b:]
p=p.replace('muse_wg5_fp8p_joint_serial','muse_wg5_fp8p_stats')
p+=r'''
_STATS_ROWS=4
@triton.jit
def _stats_merge(P,M,X,O,HQ:tl.constexpr,HKV:tl.constexpr,L:tl.constexpr,STRIDE:tl.constexpr,S:tl.constexpr,R:tl.constexpr):
    group,rb=tl.program_id(0),tl.program_id(1)
    sp=tl.arange(0,S);rows=rb*R+tl.arange(0,R);d=tl.arange(0,64)
    mx=tl.load(M+(group*S+sp[:,None])*128+rows[None,:])
    high=tl.max(mx,0);high=tl.where(high==-float('inf'),0.,high)
    w=tl.exp2(mx-high[None,:])
    sums=tl.load(M+(group*S+sp[:,None])*128+64+rows[None,:])
    den=tl.sum(w*sums,0)
    parts=tl.load(P+((group*S+sp[:,None,None])*64+rows[None,:,None])*64+d[None,None,:])
    num=tl.sum(parts*w[:,:,None],0)
    value=(num/tl.where(den>0,den,1.)[:,None]).to(tl.bfloat16).to(tl.float32)
    inv=tl.rsqrt(tl.sum(value*value,1)/64.+1.e-5)
    b=group//HKV;kh=group%HKV;token=rows//(HQ//HKV);head=kh*(HQ//HKV)+rows%(HQ//HKV)
    gate=tl.load(X+(b*L+token[:,None])*STRIDE+(HQ+2*HKV+head[:,None])*64+d[None,:]).to(tl.float32)
    tl.store(O+((b*L+token[:,None])*HQ+head[:,None])*64+d[None,:],value*inv[:,None]*tl.sigmoid(gate))
'''
ast.parse(p)
Path('dev/fp8_probability/stats_merge.cu').write_text(s)
Path('dev/ptx_campaign/attention_fp8p_stats.py').write_text(p)
