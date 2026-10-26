// HBM bandwidth probe: stream a [N][K] bf16 matrix (row-major) with cp.async.bulk rows into a smem ring.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(phase));
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes), "r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
}

// ROWS rows per stage, each CB bytes; stage = ROWS*CB. Work unit = (row block of ROWS rows) x (K split into KB-byte chunks).
template <int ROWS, int CB, int ST>
__global__ void __launch_bounds__(160) stream_kernel(const uint8_t* __restrict__ W, int N, int Kbytes, uint32_t* out, int units_per_cta_stride) {
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int STAGE = ROWS * CB;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + ST * STAGE);
  uint64_t* empty = full + ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < ST; i++) { mbar_init(&full[i], 1); mbar_init(&empty[i], 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int chunks_k = Kbytes / CB;
  const int rowblocks = N / ROWS;
  const long total_units = (long)rowblocks * chunks_k;
  uint32_t acc = 0;
  if (warp == 4) {  // producer: one warp, lane = row within stage
    int it = 0;
    for (long u = blockIdx.x; u < total_units; u += gridDim.x, it++) {
      const int s = it % ST;
      if (it >= ST) mbar_wait(&empty[s], ((it / ST) - 1) & 1);
      const long rb = u / chunks_k, kc = u % chunks_k;
      if (lane == 0) mbar_expect(&full[s], STAGE);
      __syncwarp();
      if (lane < ROWS) bulk_g2s(sm + s * STAGE + lane * CB, W + (rb * ROWS + lane) * (long)Kbytes + kc * CB, CB, &full[s]);
      // ROWS > 32 handled by multiple loops
      for (int r = lane + 32; r < ROWS; r += 32) bulk_g2s(sm + s * STAGE + r * CB, W + (rb * ROWS + r) * (long)Kbytes + kc * CB, CB, &full[s]);
    }
  } else {  // 4 consumer warps
    int it = 0;
    for (long u = blockIdx.x; u < total_units; u += gridDim.x, it++) {
      const int s = it % ST;
      mbar_wait(&full[s], (it / ST) & 1);
      const uint4* p = reinterpret_cast<const uint4*>(sm + s * STAGE);
      for (int i = tid; i < STAGE / 16; i += 128) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
    }
  }
  if (acc == 0x12345678u) out[0] = acc;
}

template <int ROWS, int CB, int ST>
void run(const uint8_t* W, int N, int K, uint32_t* out, uint8_t* flush, size_t flushn, int ctas) {
  const int smem = ST * ROWS * CB + 2 * ST * 8 + 64;
  CK(cudaFuncSetAttribute(stream_kernel<ROWS, CB, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float best = 1e9;
  for (int rep = 0; rep < 8; rep++) {
    cudaMemset(flush, rep, flushn);
    cudaEventRecord(a);
    stream_kernel<ROWS, CB, ST><<<ctas, 160, smem>>>(W, N, K * 2, out, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
  }
  CK(cudaGetLastError());
  printf("PROBE rows=%d cb=%d st=%d ctas=%d smem=%dKB: %.1f us  %.2f TB/s\n", ROWS, CB, ST, ctas, smem / 1024, best * 1000, (double)N * K * 2 / (best * 1e-3) / 1e12);
}

__global__ void __launch_bounds__(256) ldg_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  uint32_t acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride * 4) {
    uint4 a = p[i], b = i + stride < n16 ? p[i + stride] : a, c = i + 2 * stride < n16 ? p[i + 2 * stride] : a, d = i + 3 * stride < n16 ? p[i + 3 * stride] : a;
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
void run_ldg(const uint8_t* W, size_t bytes, uint32_t* out, uint8_t* flush, size_t flushn, int blocks) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float best = 1e9;
  for (int rep = 0; rep < 8; rep++) {
    cudaMemset(flush, rep, flushn);
    cudaEventRecord(a);
    ldg_kernel<<<blocks, 256>>>(reinterpret_cast<const uint4*>(W), bytes / 16, out);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
  }
  printf("LDG bytes=%zuMB blocks=%d: %.1f us %.2f TB/s\n", bytes >> 20, blocks, best * 1000, bytes / (best * 1e-3) / 1e12);
}
int main() {
  const int N = 4096, K = 8192;  // pooler: 64 MiB
  uint8_t* W; uint32_t* out; uint8_t* flush; const size_t flushn = 256u << 20;
  CK(cudaMalloc(&W, (size_t)N * K * 2)); CK(cudaMalloc(&out, 4)); CK(cudaMalloc(&flush, flushn));
  CK(cudaMemset(W, 1, (size_t)N * K * 2));
  for (int blocks : {528, 1056, 2112}) { run_ldg(W, (size_t)N * K * 2, out, flush, flushn, blocks); }
  { uint8_t* W2; CK(cudaMalloc(&W2, (size_t)2 * N * K * 2)); CK(cudaMemset(W2, 1, (size_t)2 * N * K * 2));
    for (int blocks : {1056, 2112}) { run_ldg(W2, (size_t)2 * N * K * 2, out, flush, flushn, blocks); }
    run<32, 1024, 4>(W2, 2 * N, K, out, flush, flushn, 132); run<32, 512, 4>(W2, 2 * N, K, out, flush, flushn, 264); }
  for (int ctas : {132}) {
    run<32, 512, 4>(W, N, K, out, flush, flushn, ctas);
    run<32, 1024, 4>(W, N, K, out, flush, flushn, ctas);
    run<16, 2048, 4>(W, N, K, out, flush, flushn, ctas);
    run<32, 1024, 3>(W, N, K, out, flush, flushn, ctas);
    run<16, 1024, 6>(W, N, K, out, flush, flushn, ctas);
    run<32, 512, 6>(W, N, K, out, flush, flushn, ctas);
  }
  return 0;
}
