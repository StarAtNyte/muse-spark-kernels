"""Assemble python/sglang/srt/layers/muse_spark_wm_experts.py from the kernel sources."""
import re
import sys

sys.path.insert(0, "../moe_wm")
import conv_cuda  # noqa: E402
import wm_cuda  # noqa: E402

cuda = wm_cuda.CUDA_SRC
# keep only the dispatch entries the model needs
a = cuda.index("#define WM_D(M, T, N, W_, R)")
b = cuda.index("#undef WM_ALL")
keep = """#define WM_D(M, T, N, W_, R)                                                                             \\
  if (mode == M && nt == T && nmax == N && nw == W_ && ring == R && dbg == 0) {                           \\
    launch<M, T, N, W_, R, 0>(W, S, GS, X, OUT, cnt, lst, act, nact, woff, maxslots, lmax, pdl, st);          \\
    return 0;                                                                                             \\
  }
#define WM_ALL(T, N, W_, R) WM_D(0, T, N, W_, R) WM_D(1, T, N, W_, R)
  WM_ALL(2, 1, 4, 0) WM_ALL(2, 2, 4, 0) WM_ALL(4, 4, 4, 1) WM_ALL(2, 1, 2, 0)
"""
cuda = cuda[:a] + keep + cuda[b:]
import re as _re
cuda = _re.sub(r"  WM_WG\(0, 4, 3\)[^\n]*\n", "", cuda)

# the WM_RS (row-split) macros
a = cuda.index("#define WM_RS(M, T, N, S_) \\")
b = cuda.index("#undef WM_RS\n#define WM_RSD")
rs = """#define WM_RS(M, T, N, S_) \\
  if (mode == M && ring == 4 && nt == T && nmax == N && dbg == 0) { launch_rs<M, T, N, S_, 0>(W, S, GS, X, OUT, lst, act, maxslots, lmax, pdl, st); return 0; }
  WM_RS(0, 4, 4, 4) WM_RS(1, 4, 4, 4)
"""
c = cuda.index("#undef WM_RSD")
cuda = cuda[:a] + rs + "#undef WM_RS\n" + cuda[c + len("#undef WM_RSD"):]
cuda += conv_cuda.CONV_SRC.replace("#include <cuda_runtime.h>\n#include <stdint.h>\n", "")
glue = wm_cuda.TORCH_GLUE + r"""
extern "C" void conv_raw_to_wm(const void*, const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" void conv_wm_to_marlin(const void*, const void*, void*, void*, int, int, int, cudaStream_t);
void conv_raw_to_wm_t(torch::Tensor raw, torch::Tensor sf, torch::Tensor out_w, torch::Tensor out_s) {
  conv_raw_to_wm(raw.data_ptr(), sf.data_ptr(), out_w.data_ptr(), out_s.data_ptr(), (int)raw.size(0), (int)raw.size(1), (int)raw.size(2) * 2,
                 at::cuda::getCurrentCUDAStream());
}
void conv_wm_to_marlin_t(torch::Tensor wm_w, torch::Tensor wm_s, torch::Tensor out_w, torch::Tensor out_s, int64_t N, int64_t K) {
  conv_wm_to_marlin(wm_w.data_ptr(), wm_s.data_ptr(), out_w.data_ptr(), out_s.data_ptr(), (int)wm_w.size(0), (int)N, (int)K,
                    at::cuda::getCurrentCUDAStream());
}
"""
cpp = wm_cuda.CPP_SRC + """
void conv_raw_to_wm_t(torch::Tensor raw, torch::Tensor sf, torch::Tensor out_w, torch::Tensor out_s);
void conv_wm_to_marlin_t(torch::Tensor wm_w, torch::Tensor wm_s, torch::Tensor out_w, torch::Tensor out_s, int64_t N, int64_t K);
"""
tpl = open("module_template.py").read()
out = tpl.replace("__CUDA__", cuda + glue).replace("__CPP__", cpp)
open(sys.argv[1], "w").write(out)
print("wrote", sys.argv[1], len(out))
