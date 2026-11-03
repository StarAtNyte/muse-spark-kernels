#include "fused_moe_router.cuh"
#include <torch/extension.h>

namespace muse {

void launch_fused_moe_router(
    torch::Tensor logits,
    torch::Tensor bias,
    int local_start,
    torch::Tensor out_ids,
    torch::Tensor out_weights,
    torch::Tensor out_desc_bytes)
{
  const int T = logits.size(0);
  const int num_warps = 8;
  const int block_size = num_warps * 32;
  const int grid_size = 1; // For T <= 256, 1 CTA cooperatively handles all tokens

  SmemDispatchDescriptor* desc_ptr = out_desc_bytes.defined()
      ? reinterpret_cast<SmemDispatchDescriptor*>(out_desc_bytes.data_ptr())
      : nullptr;

  int* diag_ids_ptr = out_ids.defined() ? out_ids.data_ptr<int>() : nullptr;
  float* diag_w_ptr = out_weights.defined() ? out_weights.data_ptr<float>() : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fused_moe_router_dispatch_kernel<<<grid_size, block_size, 0, stream>>>(
      logits.data_ptr<float>(),
      bias.data_ptr<float>(),
      local_start,
      T,
      desc_ptr,
      diag_ids_ptr,
      diag_w_ptr
  );
}

} // namespace muse

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_fused_moe_router", &muse::launch_fused_moe_router,
        "Launch fused MoE router, top-k selection, and dispatch descriptor initialization");
}
