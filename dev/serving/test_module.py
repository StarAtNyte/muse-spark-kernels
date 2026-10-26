"""Local (RTX 3050) test of the generated serving module: shim the extension with ctypes, check experts + Marlin conversion."""
import ctypes, os, subprocess, sys, types
import torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_moe_w4a16")); sys.path.insert(0, os.path.join(HERE, "../moe_wm"))
import reference as R
from nvfp4 import random_weights
import mq_cuda
subprocess.check_call([sys.executable, "gen_module.py", "/tmp/wm_experts_mod.py"], cwd=HERE)
src=open("/tmp/wm_experts_mod.py").read()
mod=types.ModuleType("wmx"); exec(compile(src,"wmx","exec"),mod.__dict__)
# compile kernels with clang for sm_86 and shim the extension
H=os.path.abspath(os.path.join(HERE,"../../.cuda_home/home"))
cut=mod._CUDA_SRC.index('#include <torch/extension.h>')
kern=mod._CUDA_SRC[:cut]
open(os.path.join(HERE,"srv_dev.cu"),"w").write(kern)
subprocess.check_call(["clang++","-x","cuda",f"--cuda-path={H}","--cuda-gpu-arch=sm_86","-O3","-std=c++17","-fPIC","-shared","-Wno-unknown-cuda-version","-o",os.path.join(HERE,"srv_dev.so"),os.path.join(HERE,"srv_dev.cu"),f"-L{H}/lib64","-lcudart",f"-Wl,-rpath,{H}/lib64"])
lib=ctypes.CDLL(os.path.join(HERE,"srv_dev.so")); lib.wm_set_pdl(0)
P=lambda t: ctypes.c_void_p(t.data_ptr()) if t.numel() else ctypes.c_void_p(0)
st=lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
class Shim:
    def wm_set_pdl_t(self,v): lib.wm_set_pdl(int(v))
    def wm_route(self,ids,ls,ls_val,lmax,cnt,lst,act,nact,lids,woff,grp):
        lib.wm_route_c(P(ids),ctypes.c_int(ids.numel()),P(ls),ctypes.c_int(ls_val),ctypes.c_int(lmax),P(cnt),P(lst),P(act),P(nact),P(lids),P(woff),ctypes.c_int(grp),ctypes.c_int(act.numel()//4),st())
    def wm_gemm_run(self,mode,nmax,nt,nw,ring,dbg,W,S,GS,X,OUT,cnt,lst,act,nact,woff,maxslots,lmax):
        r=lib.wm_gemm_c(*[ctypes.c_int(v) for v in (mode,nmax,nt,nw,ring,dbg)],P(W),P(S),P(GS),P(X),P(OUT),P(cnt),P(lst),P(act),P(nact),P(woff),ctypes.c_int(maxslots),ctypes.c_int(lmax),st())
        assert r==0,("cfg",mode,nmax,nt,nw,ring)
    def conv_raw_to_wm_t(self,raw,sf,ow,os_):
        lib.conv_raw_to_wm(P(raw),P(sf),P(ow),P(os_),ctypes.c_int(raw.size(0)),ctypes.c_int(raw.size(1)),ctypes.c_int(raw.size(2)*2),st())
    def conv_wm_to_marlin_t(self,ww,ws,ow,os_,N,K):
        lib.conv_wm_to_marlin(P(ww),P(ws),P(ow),P(os_),ctypes.c_int(ww.size(0)),ctypes.c_int(N),ctypes.c_int(K),st())
mod._module=Shim()
dev="cuda"; gen=torch.Generator(device=dev).manual_seed(5252)
w13,w13_sf,gs13=random_weights(8192,4096,generator=gen,device=dev,leading=(32,))
w2,w2_sf,gs2=random_weights(4096,4096,generator=gen,device=dev,leading=(32,))
weights={"w13":w13,"w13_sf":w13_sf,"w13_gs":gs13.expand(32).contiguous(),"w2":w2,"w2_sf":w2_sf,"w2_gs":gs2.expand(32).contiguous(),
         "post_norm_w":(0.1*torch.randn((4096,),generator=gen,device=dev)).to(torch.bfloat16)}
wm=mod.build_wm_experts(w13,w13_sf,weights["w13_gs"],w2,w2_sf,weights["w2_gs"])
# Marlin conversion equals the torch replica
mod.reserve_scratch(dev)
mw13,ms13,g13,mw2,ms2,g2=mod.marlin_view(wm)
rw,rs=mq_cuda.marlin_pack(w13,w13_sf)
print("marlin w13 equal",torch.equal(mw13,rw),"s13 equal",torch.equal(ms13.view(torch.uint8),rs))
rw,rs=mq_cuda.marlin_pack(w2,w2_sf)
print("marlin w2 equal",torch.equal(mw2,rw),"s2 equal",torch.equal(ms2.view(torch.uint8),rs))
for bs in (1,8,16,32,128,257):
    data=R.generate_input(bs,"ep8",seed=7+bs,dist="normal",device=dev)
    y,lids=mod.expert_outputs(wm,data["hidden_states"],data["topk_ids"],0)
    m=bs
    yf=y.float()*(1.0+weights["post_norm_w"].float())
    yf=yf*torch.rsqrt(yf.pow(2).mean(-1,keepdim=True)+R.POST_NORM_EPS)
    live=(lids>=0).float().unsqueeze(-1)
    out=(yf*data["topk_weights"].unsqueeze(-1)*live).sum(1).to(torch.bfloat16)
    ok,msg=R.check_implementation(data,out,weights)
    print(bs,ok,msg)
