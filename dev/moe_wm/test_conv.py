import ctypes, os, subprocess, sys
import torch
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
import conv_cuda, wm_cuda, mq_cuda
H=os.path.abspath(os.path.join(HERE,"../../.cuda_home/home"))
open(os.path.join(HERE,"conv_dev.cu"),"w").write(conv_cuda.CONV_SRC)
subprocess.check_call(["clang++","-x","cuda",f"--cuda-path={H}","--cuda-gpu-arch=sm_86","-O3","-std=c++17","-fPIC","-shared","-Wno-unknown-cuda-version","-o",os.path.join(HERE,"conv_dev.so"),os.path.join(HERE,"conv_dev.cu"),f"-L{H}/lib64","-lcudart",f"-Wl,-rpath,{H}/lib64"])
lib=ctypes.CDLL(os.path.join(HERE,"conv_dev.so"))
P=lambda t: ctypes.c_void_p(t.data_ptr())
st=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
dev="cuda"
torch.manual_seed(0)
for (E,R,K) in ((2,512,1024),(1,128,512),(2,256,4096)):
    raw=torch.randint(0,256,(E,R,K//2),dtype=torch.uint8,device=dev)
    # positive e4m3 scales incl. subnormals (avoid exactly zero: the wm format cannot represent S == 0)
    sf=(torch.rand(E,R,K//16,device=dev)*3+0.002).to(torch.float8_e4m3fn)
    ref_w,ref_s=wm_cuda.repack(raw,sf)
    out_w=torch.empty_like(ref_w); out_s=torch.empty_like(ref_s)
    lib.conv_raw_to_wm(P(raw),P(sf),P(out_w),P(out_s),ctypes.c_int(E),ctypes.c_int(R),ctypes.c_int(K),st)
    torch.cuda.synchronize()
    print(E,R,K,"raw->wm weights equal",torch.equal(out_w,ref_w),"scales equal",torch.equal(out_s,ref_s))
    mw,ms=mq_cuda.marlin_pack(raw,sf)
    ow=torch.empty_like(mw); os_=torch.empty_like(ms)
    lib.conv_wm_to_marlin(P(out_w),P(out_s),P(ow),P(os_),ctypes.c_int(E),ctypes.c_int(R),ctypes.c_int(K),st)
    torch.cuda.synchronize()
    print("   wm->marlin weights equal",torch.equal(ow,mw),"scales equal",torch.equal(os_,ms))
