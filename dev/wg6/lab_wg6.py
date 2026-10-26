import sys, os, importlib.util, torch, types, time
from unittest.mock import MagicMock
class _F:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sglang", "sgl_kernel"):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(); m.__path__ = []; m.__name__ = spec.name; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _F())
sys.path.insert(0, "/w/kernels/muse_attention_decode"); sys.path.insert(0, "/w/wg6")
import reference as R
import triton
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v7 = load("/w/v7/submission.py", "v7")
import wg6_cuda as W
from torch.utils.cpp_extension import load_inline
GLUE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int wg6_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,
                       int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);
void wg6_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {
  int r = wg6_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),
                  vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,
                  (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,
                  at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "wg2");
}
'''
CPP = '''void wg6_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);'''
t0 = time.time()
mod = load_inline("wg6_attn", CPP, cuda_sources=W.CUDA_SRC + GLUE, functions=["wg6_run_t"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
print("compiled", time.time() - t0, flush=True)
sys.path.insert(0, "/w/wg")
import wg_cuda as W1
GLUE1 = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int wg_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,
                      int, int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);
void wg_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
              torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t HKV, int64_t page,
              int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {
  int r = wg_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),
                 vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,
                 (int)HKV, (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,
                 at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "wg");
}
'''
CPP1 = 'void wg_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);'
mod1 = load_inline("wg1_attn", CPP1, cuda_sources=W1.CUDA_SRC + GLUE1, functions=["wg_run_t"],
                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)

def prepare(data):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    scaled = cfg['kv_scale'] != 1.0
    v7._prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](
        qkv, data['positions'], data['cache_loc'], data['k_scale'], data['v_scale'],
        q, data['k_cache'], data['v_cache'], hq, hkv, d, qkv.stride(0),
        cfg['rope_theta'], scaled, 4, num_warps=4, enable_fp_fusion=False)
    return qkv, q

def forward(data, q, qkv, splits, dbg=0, old=False):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    window = cfg['window']
    span = window + length - 1 if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 64) * 64
    partial = torch.empty((batch, hkv, splits, 64, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, 64), device=qkv.device, dtype=torch.float32)
    if old:
        mod1.wg_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                      data['v_scale'], partial, lse, length, hq, hkv, cfg['page_size'], window, splits, chunk,
                      cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), dbg)
    else:
        mod.wg6_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                      data['v_scale'], partial, lse, length, hq, cfg['page_size'], window, splits, chunk,
                      cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), dbg)
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    v7._merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)
    return result

flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
def timeit(fn, reps=12):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1); torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return ts[len(ts) // 2]

for layer, ctx in (("full", 12000), ("sliding", 6000)):
    data = R.generate_input(2, ctx, "tp8", layer, 5, q_len=8, spread="agentic", dist="normal", page=1)
    ref = R.ref_kernel(data)[0].float()
    qkv, q = prepare(data)
    for splits in (4, 8):
        out = forward(data, q, qkv, splits).float()
        err = ((out - ref).norm() / ref.norm()).item()
        o1 = forward(data, q, qkv, splits, old=True).float()
        print("WG2 chk %s splits=%d relerr_ref=%.2e relerr_old_vs_ref=%.2e new_vs_old=%.2e" % (layer, splits, err, ((o1 - ref).norm() / ref.norm()).item(), ((out - o1).norm() / o1.norm()).item()), flush=True)
    del data; torch.cuda.empty_cache()

for B in [int(x) for x in os.environ.get("BATCHES", "16,32").split(",")]:
    data = R.generate_input(B, 80000, "tp8", "full", 3, q_len=8, spread="agentic", dist="coherent", page=1)
    qkv, q = prepare(data)
    res = []
    for splits in (16, 32, 64):
        try:
            res.append("p%d=%.1f" % (splits, timeit(lambda: forward(data, q, qkv, splits))))
        except Exception as ex:
            res.append("p%d=ERR(%s)" % (splits, str(ex)[:50]))
    print("WG2 B=%d full verify" % B, " ".join(res), flush=True)
    ab = []
    for dbg in (8, 2, 16, 26):
        try:
            ab.append("d%d=%.1f" % (dbg, timeit(lambda: forward(data, q, qkv, 32, dbg))))
        except Exception as ex:
            ab.append("d%d=ERR" % dbg)
    print("WG2 ablate B=%d p33" % B, " ".join(ab), flush=True)
    del data; torch.cuda.empty_cache()
