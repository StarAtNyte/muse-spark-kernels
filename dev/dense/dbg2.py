import sys, math, torch
exec(open('test_dcomp.py').read().split("torch.manual_seed(0)")[0])
torch.manual_seed(1)
N,K=512,1024
w=(torch.randn(N,K,device=dev)/math.sqrt(K)).to(torch.bfloat16)
tiles,emax,eoff,esc=D.dense_crepack(w)
rows,cols=esc[:,0].long(),esc[:,1].long()
print("escapes",len(rows))
bad=0
for ks in (1,2):
  for idx in range(0,len(rows),max(1,len(rows)//12)):
    k=int(cols[idx]); x=torch.zeros(8,K,device=dev,dtype=torch.bfloat16); x[0,k]=1.0
    out=torch.empty(8,N,dtype=torch.bfloat16,device=dev); part=torch.empty(ks,8,N,dtype=torch.float32,device=dev); cnt=torch.zeros(4096,dtype=torch.int32,device=dev)
    om=1 if ks==1 else 0
    r=lib.dense_crun(*[ctypes.c_int(v) for v in (1,1,4,4,om)],P(tiles),P(x),P(out) if om==1 else P(part),ctypes.c_int(8),ctypes.c_int(N),ctypes.c_int(K),ctypes.c_int(ks),P(cnt),P(out),ctypes.c_int(N),ctypes.c_int(emax),P(eoff),P(esc),stream())
    if om==0: lib.dense_reduce(P(part),P(out),ctypes.c_int(8),ctypes.c_int(N),ctypes.c_int(N),ctypes.c_int(ks),ctypes.c_int(0),ctypes.c_int(0),stream())
    torch.cuda.synchronize()
    if not torch.equal(out[0], w[:,k]): bad+=1; print("mismatch ks",ks,idx)
print("bad",bad)
