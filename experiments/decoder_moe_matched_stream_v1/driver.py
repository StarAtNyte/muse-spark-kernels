"""Decoder MoE floor diagnostic: per decode/verify shape, local routes, distinct experts, layer time,
isolated moe_fused time (cold L2) and pure-read floor of the active experts' weights."""
import importlib.util, json, statistics, sys, os
from unittest.mock import MagicMock
import torch
class StubSGLang:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('sglang', 'sgl_kernel'):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, StubSGLang())
sys.path.insert(0, '/w/kernels/muse_decode_layer')
import reference as R
spec = importlib.util.spec_from_file_location('sub', '/w/run/sub.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


from torch.utils.cpp_extension import load_inline
CU = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
struct Seg { const uint4* p; long long n; };
struct Segs { Seg s[64]; int k; };
__global__ void __launch_bounds__(512) stream_k(Segs S, unsigned* out) {
  unsigned acc = 0;
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long long)gridDim.x * blockDim.x;
  for (int j = 0; j < S.k; ++j) {
    const uint4* p = S.s[j].p; long long n = S.s[j].n;
    long long i = tid;
    for (; i + 3 * nth < n; i += 4 * nth) {
      uint4 a, b, c, d;
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p + i));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(p + i + nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(p + i + 2 * nth));
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "l"(p + i + 3 * nth));
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += nth) { uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
__global__ void empty_k(unsigned* out) { if (threadIdx.x == 1023) out[0] = 1; }
void run_stream(std::vector<torch::Tensor> ts, torch::Tensor out, int grid, int threads) {
  Segs S; S.k = (int)ts.size();
  for (int j = 0; j < S.k; ++j) { S.s[j].p = (const uint4*)ts[j].data_ptr(); S.s[j].n = ts[j].numel() * ts[j].element_size() / 16; }
  stream_k<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(S, (unsigned*)out.data_ptr());
  cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
void run_empty(torch::Tensor out) { empty_k<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((unsigned*)out.data_ptr()); }
'''
ext = load_inline('stream_floor', cpp_sources='void run_stream(std::vector<torch::Tensor> ts, torch::Tensor out, int grid, int threads); void run_empty(torch::Tensor out);',
                  cuda_sources=CU, functions=['run_stream', 'run_empty'], extra_cuda_cflags=['-O3', '-arch=sm_90a'])

rec = []
orig = S._moe
class Proxy:
    def __init__(self, e): self.e = e
    def __getattr__(self, k): return getattr(self.e, k)
    def okc_run(self, *args):
        rec.append((self.e, args))
        return self.e.okc_run(*args)
def proxy(M=None, state=None):
    return Proxy(orig(M, state))
S._moe = proxy
w = R.generate_weights(R.WEIGHTS_SEED)
st = S.prepare(w)
flush = torch.zeros(256 * 1024 * 1024 // 4, device='cuda')
outb = torch.zeros(4, dtype=torch.int32, device='cuda')
def cap(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); return g
def measure(gs, n=20):
    times = {k: [] for k in gs}
    for rep in range(n):
        for k in (list(gs) if rep % 2 == 0 else list(reversed(gs))):
            flush.sum(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); gs[k].replay(); b.record(); b.synchronize(); times[k].append(a.elapsed_time(b)*1000)
    return {k: statistics.median(v) for k,v in times.items()}
for batch in (16,32):
    for seed in (4001,4002):
        c=dict(batch=batch,ctx=80000,layer='sliding',page=1,dist='soft',q_len=1,spread='agentic',seed=seed)
        ls=R.case_fields(c,w)['local_start']; d=R.generate_input(**c,local_start=ls)
        rec.clear(); o=S.kernel_fn(d,st); torch.cuda.synchronize()
        ok,msg=R.check_implementation(d,R.collect(d,o),w)
        if not rec: raise RuntimeError('no MoE launch captured')
        e,a=rec[-1]; ids=a[3]; active=ids[(ids>=0)&(ids<32)].unique().tolist()
        # M=16 uses skinny units (10/11); M=32 uses WGS stages (12/13).
        wi=(10,11) if batch <= 16 else (12,13)
        segs=[]
        for i in active:
            for ix in wi:
                t=a[0][ix]; n=t.numel()//32; segs.append(t.narrow(0,i*n,n))
        nb=sum(t.numel()*t.element_size() for t in segs)
        gs={'moe':cap(lambda:e.okc_run(*a)), 'stream264':cap(lambda:ext.run_stream(segs,outb,264,512)), 'stream528':cap(lambda:ext.run_stream(segs,outb,528,256))}
        times=measure(gs)
        print('ROW',json.dumps(dict(batch=batch,seed=seed,active=active,bytes=nb,reference_ok=ok,reference_msg=msg[:150],us=times,stream_over_moe=min(times['stream264'],times['stream528'])/times['moe'])),flush=True)
print('DONE',flush=True)
