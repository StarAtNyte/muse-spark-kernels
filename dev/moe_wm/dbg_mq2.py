import sys, os, ctypes, subprocess, time
import torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_moe_w4a16"))
from nvfp4 import random_weights, dequantize
import mq_cuda
H=os.path.abspath(os.path.join(HERE,"../../.cuda_home/home"))
open(os.path.join(HERE,"mq_dev.cu"),"w").write(mq_cuda.CUDA_SRC)
subprocess.check_call(["clang++","-x","cuda",f"--cuda-path={H}","--cuda-gpu-arch=sm_86","-O3","-std=c++17","-fPIC","-shared","-Wno-unknown-cuda-version","-o",os.path.join(HERE,"mq_dev.so"),os.path.join(HERE,"mq_dev.cu"),f"-L{H}/lib64","-lcudart",f"-Wl,-rpath,{H}/lib64"])
lib=ctypes.CDLL(os.path.join(HERE,"mq_dev.so")); lib.wm_set_pdl(0)
P=lambda t: ctypes.c_void_p(t.data_ptr())
dev="cuda"; gen=torch.Generator(device=dev).manual_seed(1)
E=2
w2,w2_sf,gs2=random_weights(4096,4096,generator=gen,device=dev,leading=(E,))
W,S=mq_cuda.marlin_pack(w2,w2_sf)
print(W.shape,S.shape)
gs=gs2.expand(E).contiguous()
stored=(gs.float()*2.0**119).to(torch.bfloat16).float(); GE=(stored*2.0**-111).contiguous()
# one expert=0 with 1 token: lst=[0]; act: item (0,0,1,1)

sf1=torch.ones_like(w2_sf.float()).to(torch.float8_e4m3fn)
W,S=mq_cuda.marlin_pack(w2,sf1)
GE1=torch.full((E,),1.0,device=dev)*(2.0**119*1.0)
GE1=(GE1.to(torch.bfloat16).float()*2.0**-111).contiguous()
from nvfp4 import unpack_e2m1
vals=unpack_e2m1(w2[0])  # [4096,4096] float values (e2m1)
def run(kidx):
    x=torch.zeros(8,4096,device=dev,dtype=torch.bfloat16); x[0,kidx]=1.0
    lst=torch.zeros(32*8,dtype=torch.int32,device=dev); lst[0]=0
    act=torch.zeros(512,dtype=torch.int32,device=dev); act[0:4]=torch.tensor([0,0,1,1],dtype=torch.int32,device=dev)
    out=torch.zeros(8,4096,dtype=torch.bfloat16,device=dev)
    r=lib.mq_gemm_c(ctypes.c_int(1),ctypes.c_int(1),ctypes.c_int(4),P(W),P(S),P(GE1),P(x),P(out),P(lst),P(act),ctypes.c_int(1),ctypes.c_int(8),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return out[0].float(), vals[:,kidx].float()
for k in (0,1,2,3,8,9,15,16,17,64):
    o,ref=run(k)
    ok=(o==ref).float().mean().item()
    # try to find where each ref row landed: check if o equals permuted ref
    print("k",k,"match frac",round(ok,3), "o[:16]",o[:16].tolist(), "ref[:16]",ref[:16].tolist())
