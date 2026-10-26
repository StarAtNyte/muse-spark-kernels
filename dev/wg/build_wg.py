import os, sys
import wg_cuda as W
sys.path.insert(0, "../attn")

base = open("../../experiments/attention_fp8_hybrid_v7/submission.py").read()
base = base.replace("def kernel_fn(data):", "def _v7_kernel_fn(data):", 1)
lab_attn = open("../attn/lab_attn.py").read()
lab_data = lab_attn.split("def _fd_check():", 1)[0]          # _lab_data
glue_cuda = r'''
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
cpp = '''
void wg_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
              torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t HKV, int64_t page,
              int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);
'''
glue = r'''
_WG_MOD = None


def _wg_module():
    global _WG_MOD
    if _WG_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WG_MOD = load_inline("muse_wg_attn", _WG_CPP, cuda_sources=_WG_CUDA, functions=["wg_run_t"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _WG_MOD


def _wg_prepare(data):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=qkv.device)
    scaled = cfg['kv_scale'] != 1.0
    _prepare_qkv[(tokens, triton.cdiv(hq + hkv, 4))](
        qkv, data['positions'], data['cache_loc'], data['k_scale'], data['v_scale'],
        q, data['k_cache'], data['v_cache'], hq, hkv, d, qkv.stride(0),
        cfg['rope_theta'], scaled, 4, num_warps=4, enable_fp_fusion=False)
    return qkv, q


def _wg_forward(data, q, qkv, splits, dbg=0):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    window = cfg['window']
    span = window + length - 1 if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 128) * 128
    partial = torch.empty((batch, hkv, splits, 64, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, 64), device=qkv.device, dtype=torch.float32)
    _wg_module().wg_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                          data['v_scale'], partial, lse, length, hq, hkv, cfg['page_size'], window, splits, chunk,
                          cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), dbg)
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)
    return result


_WG_LAB = True
_WG_POLICY = __WGPOL__


def kernel_fn(data):
    cfg = data['config']
    if _WG_LAB and not torch.cuda.is_current_stream_capturing() and not getattr(kernel_fn, '_labbed', False):
        kernel_fn._labbed = True
        _wg_lab()
    return _v7_kernel_fn(data)


def _wg_lab():
    flush = torch.zeros(160 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')

    def timeit(fn, reps=10):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(reps):
            flush.add_(1)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        ts.sort()
        return ts[len(ts) // 2]

    for batch, ctx, layer in ((2, 12000, 'full'), (2, 6000, 'sliding'), (1, 300, 'sliding')):
        data = _lab_data(batch, 8, layer, 'agentic', ctx=ctx, seed=11 + batch)
        ref = _v7_kernel_fn(data).float()
        qkv, q = _wg_prepare(data)
        for splits, dbgv in ((8, 0), (8, 48)):
            try:
                out = _wg_forward(data, q, qkv, splits, dbgv).float()
                err = ((out - ref).norm() / ref.norm()).item()
                print('LABW chk B=%d %s splits=%d dbg=%d relerr_vs_v7=%.2e finite=%d' % (batch, layer, splits, dbgv, err, int(torch.isfinite(out).all())), flush=True)
            except Exception as ex:
                print('LABW chk ERR', str(ex)[:120], flush=True)
        del data
        torch.cuda.empty_cache()
    for batch, layer, spread in ((1, 'sliding', 'fixed'), (1, 'full', 'fixed'), (16, 'sliding', 'agentic'), (16, 'full', 'agentic'),
                                 (32, 'sliding', 'agentic'), (32, 'full', 'agentic')):
        data = _lab_data(batch, 8, layer, spread, seed=batch + 3)
        qkv, q = _wg_prepare(data)
        ref = timeit(lambda: _v7_kernel_fn(data))
        res = []
        for splits in ((2, 4, 8, 16) if layer == 'sliding' else (16, 32, 64, 128)):
            try:
                res.append((timeit(lambda: _wg_forward(data, q, qkv, splits)), splits))
            except Exception as ex:
                res.append((-1, splits))
        bs = sorted(r for r in res if r[0] > 0)
        if bs:
            sp = bs[0][1]
            ab = []
            for dbg in (1, 2, 4, 8, 15):
                try:
                    ab.append('d%d=%.1f' % (dbg, timeit(lambda: _wg_forward(data, q, qkv, sp, dbg))))
                except Exception as ex:
                    ab.append('d%d=ERR' % dbg)
            print('LABW ablate B=%d %s p%d' % (batch, layer, sp), ' '.join(ab), flush=True)
        print('LABW B=%d L=8 %s v7=%.1f' % (batch, layer, ref), ' '.join('%.1f:p%d' % r for r in sorted(r for r in res if r[0] > 0)), flush=True)
        del data
        torch.cuda.empty_cache()
'''
out = base + "\n\n" + lab_data + "\n_WG_CUDA = r'''" + W.CUDA_SRC + glue_cuda + "'''\n_WG_CPP = r'''" + cpp + "'''\n" + glue.replace("__WGPOL__", "{}")
open(sys.argv[1], "w").write(out)
print("wrote")
