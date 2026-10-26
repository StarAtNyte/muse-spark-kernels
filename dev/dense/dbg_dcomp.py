import sys, math, torch
sys.argv=['x']
exec(open('test_dcomp.py').read().split("torch.manual_seed(0)")[0])
torch.manual_seed(0)
N,K=512,1024
w=(torch.randn(N,K,device=dev)/math.sqrt(K)).to(torch.bfloat16)
tiles,emax,eoff,esc=D.dense_crepack(w)
x=torch.randn(8,K,device=dev).to(torch.bfloat16)
out=torch.full((8,N),7.0,dtype=torch.bfloat16,device=dev)
part=torch.empty(1,8,N,dtype=torch.float32,device=dev); cnt=torch.zeros(4096,dtype=torch.int32,device=dev)
r=lib.dense_crun(*[ctypes.c_int(v) for v in (1,1,4,4,1)],P(tiles),P(x),P(out),ctypes.c_int(8),ctypes.c_int(N),ctypes.c_int(K),ctypes.c_int(1),P(cnt),P(out),ctypes.c_int(N),ctypes.c_int(emax),P(eoff),P(esc),stream())
print("ret",r, torch.cuda.synchronize())
ref=x.double()@w.double().T
print(out[0,:6], ref[0,:6])
