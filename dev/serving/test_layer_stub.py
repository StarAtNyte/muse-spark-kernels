"""Run the patched MuseSpark1xFp4Experts control flow locally with stubbed SGLang dependencies (RTX 3050)."""
import ctypes, enum, importlib.util, os, subprocess, sys, types
import torch
HERE=os.path.dirname(os.path.abspath(__file__))
SRV=os.path.join(HERE,"../../serving/sglang/python/sglang")
sys.path.insert(0, os.path.join(HERE, "../../kernels/muse_moe_w4a16"))
import reference as R
from nvfp4 import random_weights

def mod(name, **attrs):
    m=types.ModuleType(name); m.__path__=[]
    for k,v in attrs.items(): setattr(m,k,v)
    sys.modules[name]=m
    return m
def load_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

class Backend(enum.Enum):
    MARLIN="marlin"; AUTO="auto"; FLASHINFER_TRTLLM="trtllm"
    def is_marlin(self): return self is Backend.MARLIN
    def is_auto(self): return self is Backend.AUTO
    def is_flashinfer_trtllm(self): return False
    def is_flashinfer_trtllm_routed(self): return False
class Cfg:
    def __init__(self, **kw): self.__dict__.update(kw)
class Base:
    def __init__(self, quant_config): self.quant_config=quant_config
class QuantCfg: group_size=16
class Norm(torch.nn.Module):
    def __init__(self, w): super().__init__(); self.weight=w; self.eps=1e-8
    def _get_effective_weight(self): return self.weight
for n in ("sglang","sglang.kernels","sglang.kernels.ops","sglang.kernels.ops.muse_spark_v12","sglang.srt","sglang.srt.layers","sglang.srt.layers.moe","sglang.srt.layers.moe.moe_runner",
          "sglang.srt.layers.quantization","sglang.srt.utils"): mod(n)
mod("sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm", FusedPostExpertNormWeightedSumTRTLLM=object)
mod("sglang.kernels.ops.muse_spark_v12.norm_fp4_quant", fused_norm_fp4_quant=None)
mod("sglang.srt.distributed", get_moe_tp_group=None)
mod("sglang.srt.layers.moe", MoeRunnerBackend=Backend, get_moe_runner_backend=lambda: Backend.MARLIN)
mod("sglang.srt.layers.moe.moe_runner.base", MoeRunnerConfig=Cfg)
mod("sglang.srt.layers.quantization.base_config", QuantizationConfig=object)
mod("sglang.srt.utils.common", get_device_capability=lambda: torch.cuda.get_device_capability(), is_cuda=lambda: True)
mod("sglang.srt.layers.muse_spark_layers", MuseSpark1xRMSNormWithInputScale=Norm)
mod("sglang.srt.layers.muse_spark_expert_tp", partition=lambda i: (i, False))
mod("sglang.srt.layers.utils", copy_or_rebind_param=lambda l,n,v: setattr(l,n,v))
mod("sglang.srt.layers.quantization.modelopt_quant", ModelOptNvFp4FusedMoEMethod=Base, _compute_gemm1_alphas=None)
load_file("sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton", SRV+"/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py")

# the generated wm module, with the kernel extension shimmed through ctypes/clang for sm_86
subprocess.check_call([sys.executable,"gen_module.py","/tmp/wm_experts_mod.py"],cwd=HERE)
wmx=load_file("sglang.srt.layers.muse_spark_wm_experts","/tmp/wm_experts_mod.py")
sys.modules["sglang.srt.layers"].muse_spark_wm_experts=wmx
H=os.path.abspath(os.path.join(HERE,"../../.cuda_home/home"))
open(os.path.join(HERE,"srv_dev.cu"),"w").write(wmx._CUDA_SRC[:wmx._CUDA_SRC.index('#include <torch/extension.h>')])
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
wmx._module=Shim()
wmx._SM_MAJOR=torch.cuda.get_device_capability()[0]
# only a stubbed Marlin kernel is available for the long-batch branch: record that it is reached with the converted tensors
# stubs for the stock Marlin branch (long batches): record the tensors it is given
calls=[]
def fake_gemm(*a,**k):
    calls.append((a[2].shape,a[4].dtype,a[5].shape)); return torch.zeros((a[0].shape[0]*(8 if k.get('top_k')==8 else 1)//1 if False else 1,1))
mod("sgl_kernel"); mod("sgl_kernel.scalar_type", scalar_types=types.SimpleNamespace(float4_e2m1f=None))
mod("sglang.kernels.ops.activation"); mod("sglang.kernels.ops.activation.activation", silu_and_mul=lambda *a,**k: None)
mod("sglang.kernels.ops.moe"); mod("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=fake_gemm)
mod("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=lambda *a,**k:(None,None,None))
mod("sglang.srt.layers.quantization.marlin_utils", marlin_make_workspace=lambda *a,**k: None)
load_file("sglang.srt.layers.quantization.muse_spark_nvfp4_moe", SRV+"/srt/layers/quantization/muse_spark_nvfp4_moe.py")
E=load_file("sglang.srt.layers.muse_spark_fp4_moe", SRV+"/srt/layers/muse_spark_fp4_moe.py")

dev="cuda"; gen=torch.Generator(device=dev).manual_seed(5252)
w13,w13_sf,gs13=random_weights(8192,4096,generator=gen,device=dev,leading=(32,))
w2,w2_sf,gs2=random_weights(4096,4096,generator=gen,device=dev,leading=(32,))
post_w=(0.1*torch.randn((4096,),generator=gen,device=dev)).to(torch.bfloat16)
weights={"w13":w13,"w13_sf":w13_sf,"w13_gs":gs13.expand(32).contiguous(),"w2":w2,"w2_sf":w2_sf,"w2_gs":gs2.expand(32).contiguous(),"post_norm_w":post_w}
layer=E.MuseSpark1xFp4Experts(num_experts=256,num_local_experts=32,local_expert_offset=0,hidden_size=4096,intermediate_size=4096,top_k=8,quant_config=QuantCfg(),params_dtype=torch.bfloat16)
layer.gate_up_proj.data=w13.clone(); layer.gate_up_proj_weight_scale.data=w13_sf.clone()
layer.gate_up_proj_weight_scale_2.data=weights["w13_gs"].view(32,1).expand(32,2).contiguous()
layer.down_proj.data=w2.clone(); layer.down_proj_weight_scale.data=w2_sf.clone(); layer.down_proj_weight_scale_2.data=weights["w2_gs"].clone()
print("supports:",wmx.supports(layer))
layer.quant_method.process_weights_after_loading(layer)
print("wm built:",layer._wm_experts is not None,"params left:",[n for n,_ in layer.named_parameters()])
norm=Norm((post_w.float()+1.0).to(torch.bfloat16))
class WN(Norm): pass
for bs in (1,8,16,32,128,257):
    data=R.generate_input(bs,"ep8",seed=7+bs,dist="normal",device=dev)
    out=layer._forward_marlin(data["hidden_states"],data["topk_weights"],data["topk_ids"],norm)
    ok,msg=R.check_implementation(data,out,weights)
    print(bs,ok,msg)
# the long-batch branch: only up to the converted tensors (no Marlin kernel here)
data=R.generate_input(400,"ep8",seed=3,dist="normal",device=dev)
try:
    layer._forward_marlin(data["hidden_states"],data["topk_weights"],data["topk_ids"],norm)
    print("marlin calls recorded:",calls[:2])
except Exception as ex:
    print("long batch reached stock path (expected failure without sgl_kernel):",type(ex).__name__, str(ex)[:80])
print("scratch bound:",type(layer.w13_weight).__name__, tuple(layer.w13_weight.shape), layer.w13_weight_scale.dtype, layer.w13_weight_scale_2.dtype, tuple(layer.w13_weight_scale_2.shape))
