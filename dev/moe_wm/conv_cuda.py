"""Layout conversion kernels for the serving patch: raw NVFP4 -> weight-major (wm) layout at load, wm -> Marlin layout at run time."""

CONV_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

// bit sources of the four code bits (m0, m1, m2, sign) for register j = rh + 2 p, low half (the high half is +16)
__constant__ int c_src[4][4] = {{6, 7, 8, 15}, {9, 10, 11, 18}, {12, 13, 14, 21}, {4, 0, 1, 3}};

__device__ __forceinline__ int wm_pos(int j, int bit, int h) { return (c_src[j][bit] + 16 * h) & 31; }

// e4m3 scale -> b' with bf16 bits of S * 2^118 == (b' << 4) + 0x7600 (clamped at the smallest e4m3 subnormal)
__device__ __forceinline__ uint8_t s2bp(uint8_t s8) {
  const int sign_exp = s8 & 0x7F;                       // scales are non-negative
  float f;
  {
    const int e = (sign_exp >> 3) & 15, m = sign_exp & 7;
    f = e == 0 ? (m * 0.001953125f) : (1.0f + m * 0.125f) * __int_as_float((e - 7 + 127) << 23);
  }
  f *= __int_as_float((118 + 127) << 23);               // 2^118
  int bits = __float_as_uint(f) >> 16;
  if (bits < 0x7600) bits = 0x7600;
  return (uint8_t)((bits - 0x7600) >> 4);
}

// raw: uint8 [E, R, K/2] (low nibble = even k), sf: e4m3 [E, R, K/16]  ->  words int32 [E, RTn, KCn, 32, 4], scales uint8 [E, RTn, KCn/4, 32, 8]
__global__ void raw_to_wm_w(const uint8_t* __restrict__ raw, uint32_t* __restrict__ out, int R, int K) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int RTn = R / 16, KCn = K / 64;
  const long total = (long)gridDim.y * RTn * KCn * 128;
  if (idx >= total) return;
  const int ks = idx & 3, lane = (idx >> 2) & 31;
  long r = idx >> 7;
  const int kc = r % KCn; r /= KCn;
  const int rt = r % RTn;
  const int e = blockIdx.y;
  const int g = lane >> 2, t = lane & 3;
  uint32_t word = 0;
#pragma unroll
  for (int rh = 0; rh < 2; rh++)
#pragma unroll
    for (int p = 0; p < 2; p++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int row = rt * 16 + rh * 8 + g;
        const int k = kc * 64 + t * 16 + ks * 4 + p * 2 + h;
        const uint8_t byte = raw[((size_t)e * R + row) * (K / 2) + (k >> 1)];
        const uint32_t code = (k & 1) ? (byte >> 4) : (byte & 15);
        const int j = rh + 2 * p;
#pragma unroll
        for (int bit = 0; bit < 4; bit++) word |= ((code >> bit) & 1u) << wm_pos(j, bit, h);
      }
  out[(((size_t)e * RTn + rt) * KCn + kc) * 128 + lane * 4 + ks] = word;
}

__global__ void raw_to_wm_s(const uint8_t* __restrict__ sf, uint8_t* __restrict__ out, int R, int K) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const int RTn = R / 16, KCn = K / 64, KGn = K / 16;
  const long total = (long)gridDim.y * RTn * (KCn / 4) * 256;
  if (idx >= total) return;
  const int b = idx & 7, lane = (idx >> 3) & 31;
  long r = idx >> 8;
  const int kcg = r % (KCn / 4); r /= (KCn / 4);
  const int rt = r % RTn;
  const int e = blockIdx.y;
  const int rh = b >> 2, c = b & 3, g = lane >> 2, t = lane & 3;
  const int row = rt * 16 + rh * 8 + g, grp = kcg * 16 + c * 4 + t;
  out[(size_t)e * RTn * (KCn / 4) * 256 + ((((size_t)rt * (KCn / 4) + kcg) * 32 + lane) * 8 + b)] = s2bp(sf[((size_t)e * R + row) * KGn + grp]);
}

// wm -> Marlin: words int32 [E, K/16, N/64*128], scales uint8 [E, K/16, N]
__global__ void wm_to_marlin_w(const uint32_t* __restrict__ wm, uint32_t* __restrict__ out, int N, int K) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // output word index within the expert
  const long per = (long)(K / 16) * (N / 64) * 128;
  if (idx >= per) return;
  const int RTn = N / 16, KCn = K / 64;
  const int e = blockIdx.y;
  const int wq = idx & 3, lane = (idx >> 2) & 31;
  long r = idx >> 7;
  const int nt = r % (N / 64); r /= (N / 64);
  const int kt = (int)r;
  const int g = lane >> 2, tq = lane & 3;
  uint32_t res = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int kl = i >> 2, hn = (i >> 1) & 1, kh = i & 1;
    const int n = nt * 64 + wq * 16 + hn * 8 + g;
    const int k = kt * 16 + kh * 8 + tq * 2 + kl;
    const int rt = n >> 4, rh = (n >> 3) & 1, gg = n & 7;
    const int kc = k >> 6, rr = k & 63;
    const int t = rr >> 4, ks = (rr & 15) >> 2, p = (rr & 3) >> 1, h = rr & 1;
    const uint32_t w = wm[(((size_t)e * RTn + rt) * KCn + kc) * 128 + (gg * 4 + t) * 4 + ks];
    const int j = rh + 2 * p;
    uint32_t code = 0;
#pragma unroll
    for (int bit = 0; bit < 4; bit++) code |= ((w >> wm_pos(j, bit, h)) & 1u) << bit;
    res |= code << (4 * i);
  }
  out[(size_t)e * per + idx] = res;
}

__global__ void wm_to_marlin_s(const uint8_t* __restrict__ wms, uint8_t* __restrict__ out, int N, int K) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;      // (kg, pos) within the expert
  const long per = (long)(K / 16) * N;
  if (idx >= per) return;
  const int RTn = N / 16, KCn = K / 64;
  const int e = blockIdx.y;
  const int kg = idx / N, pos = idx % N;
  const int blk = pos >> 6, p = pos & 63;
  const int p0 = (p & ~3) | ((p & 3) == 1 ? 2 : (p & 3) == 2 ? 1 : (p & 3));   // swap of the middle pair of each 4
  const int nl = (p0 >> 3) + 8 * (p0 & 7);
  const int n = blk * 64 + nl;
  const int rt = n >> 4, rh = (n >> 3) & 1, g = n & 7;
  const int kcg = kg >> 4, c = (kg >> 2) & 3, t = kg & 3;
  const uint8_t bp = wms[(size_t)e * RTn * (KCn / 4) * 256 + ((((size_t)rt * (KCn / 4) + kcg) * 32 + (g * 4 + t)) * 8 + rh * 4 + c)];
  out[(size_t)e * per + idx] = (uint8_t)(bp + 104);
}

extern "C" void conv_raw_to_wm(const void* raw_w, const void* sf, void* out_w, void* out_s, int E, int R, int K, cudaStream_t st) {
  {
    const long total = (long)(R / 16) * (K / 64) * 128;
    dim3 grid((total + 255) / 256, E);
    raw_to_wm_w<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(raw_w), reinterpret_cast<uint32_t*>(out_w), R, K);
  }
  {
    const long total = (long)(R / 16) * (K / 256) * 256;
    dim3 grid((total + 255) / 256, E);
    raw_to_wm_s<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(sf), reinterpret_cast<uint8_t*>(out_s), R, K);
  }
}

extern "C" void conv_wm_to_marlin(const void* wm_w, const void* wm_s, void* out_w, void* out_s, int E, int N, int K, cudaStream_t st) {
  {
    const long per = (long)(K / 16) * (N / 64) * 128;
    dim3 grid((per + 255) / 256, E);
    wm_to_marlin_w<<<grid, 256, 0, st>>>(reinterpret_cast<const uint32_t*>(wm_w), reinterpret_cast<uint32_t*>(out_w), N, K);
  }
  {
    const long per = (long)(K / 16) * N;
    dim3 grid((per + 255) / 256, E);
    wm_to_marlin_s<<<grid, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(wm_s), reinterpret_cast<uint8_t*>(out_s), N, K);
  }
}
"""
