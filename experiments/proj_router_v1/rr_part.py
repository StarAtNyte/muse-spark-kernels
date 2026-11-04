

# + proj_router_v1 (2026-11-05): FP32 router through "rr", a generalisation of vjeux's rk kernel (same TMA + SS-wgmma
# hi + lo scheme, same in-kernel clear with a grid counter, same order-dependent FP32 red.add.v4 split-K combine) with
# the tile as a template: WG consumer warpgroups (64 W rows each, W the A operand) x T tokens (the wgmma N, 8..256), a
# QS-deep TMA ring, and S K-splits chosen per m so the grid covers the GPU. Each product is x * hi + x * lo (x is BF16
# exactly, hi / lo the BF16 split of the FP32 weights) with FP32 accumulation: relative error ~2^-17 per weight.
# prepare() validates each route against an FP64 reference (random, outlier and one-hot probe rows) and falls back to
# the dispatcher on any failure.
_RR_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define RD __device__ __forceinline__
RD uint64_t rr_desc(uint32_t a) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
RD void rr_init(uint32_t b, uint32_t c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c)); }
RD void rr_expect(uint32_t b, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
RD void rr_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
RD void rr_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRRW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RRW_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
RD void rr_tma(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
template<int N> RD void rr_mma(float* d, uint64_t da, uint64_t db);
@@MMA@@

RD void rr_tma_mc(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
RD void rr_csync() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }

// CL > 1: a cluster of CL CTAs = CL consecutive token tiles of the same (W rows, K split); each CTA loads 1/CL of the
// W hi | lo stage and multicasts it to the whole cluster (W read from L2 once per cluster, not once per token tile);
// every consumer warpgroup releases a stage on every CTA's empty barrier. ROT: the red.add order starts at a K-split-
// dependent offset so the S partial tiles of one output tile do not hit the same addresses at the same time.
template<int WG, int T, int QS, int CL, int ROT>
__global__ void __launch_bounds__(WG * 128 + 64, 1) rr_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                              float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int R = 64 * WG, NRT = 256 / R, WT = R * 128, ST = 2 * WT + T * 128, NA = T / 2, NC = WG * 128, TP = R + 4;
  constexpr int HP = CL > 1 ? CL / 2 : 1, PR = R / HP;   // W pieces per half (hi / lo) and rows per piece
  extern __shared__ __align__(1024) unsigned char rr_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rr_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int tt = blockIdx.x % ntt, rt = (blockIdx.x / ntt) % NRT, sp = blockIdx.x / (ntt * NRT);
  const bool Z = S != 1;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t rank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rr_init(fullb + 8 * i, 1); rr_init(emptyb + 8 * i, WG * CL); }
    rr_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (CL > 1) rr_csync(); else __syncthreads();
  if (warp == WG * 4 + 1) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    if (Z) {
      int target = 0;
      if (lane == 0) {
        int v;
        asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
      }
      const int n4 = M * 64;
      const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
      const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
      for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      __syncwarp();
      if (lane == 0) {
        asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        for (;;) {
          int v;
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
          if (v - target >= 0) break;
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
          if (t1 - t0 > 2000000000ull) __trap();
          __nanosleep(64);
        }
        rr_arrive(zbar);
      }
      __syncwarp();
    }
  } else if (warp == WG * 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rr_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rr_expect(fb, ST);
        const int kb = kb0 + i;
        if constexpr (CL == 1) {
          rr_tma(st, &mapW, fb, kb * 64, rt * R);
          rr_tma(st + WT, &mapW, fb, 8192 + kb * 64, rt * R);
        } else {
          const int half = (int)rank / HP, pc = (int)rank % HP;
          rr_tma_mc(st + half * WT + pc * PR * 128, &mapW, fb, half * 8192 + kb * 64, rt * R + pc * PR, (uint16_t)((1u << CL) - 1));
        }
        rr_tma(st + 2 * WT, &mapX, fb, kb * 64, tt * T);
      }
    }
    __syncwarp();
  } else {
    const int wg = tid >> 7, lt = tid & 127;
    float d[NA];
#pragma unroll
    for (int j = 0; j < NA; ++j) d[j] = 0.f;
    for (int i = 0; i < nk; ++i) {
      const int s = i % QS;
      rr_wait(fullb + 8 * s, (i / QS) & 1);
#pragma unroll
      for (int j = 0; j < NA; ++j) asm volatile("" : "+f"(d[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      const uint32_t a = sbase + s * ST + wg * 8192, b = sbase + s * ST + 2 * WT;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        rr_mma<T>(d, rr_desc(a + 32 * k16), rr_desc(b + 32 * k16));
        rr_mma<T>(d, rr_desc(a + WT + 32 * k16), rr_desc(b + 32 * k16));
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
      for (int j = 0; j < NA; ++j) asm volatile("" : "+f"(d[j]) :: "memory");
      if (i > 0 && lt == 0) {
        const uint32_t eb = emptyb + 8 * ((i - 1) % QS);
        if constexpr (CL == 1) rr_arrive(eb);
        else {
#pragma unroll
          for (int q = 0; q < CL; ++q) {
            uint32_t ra;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(eb), "r"(q));
            asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(ra) : "memory");
          }
        }
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < NA; ++j) asm volatile("" : "+f"(d[j]) :: "memory");
    asm volatile("bar.sync 1, %0;" :: "n"(NC) : "memory");   // every warpgroup's wgmmas retired: the ring is free
    float* Ts = reinterpret_cast<float*>(rr_smem + (sbase - sraw));
    const int row = wg * 64 + (lt >> 5) * 16 + ((lt & 31) >> 2);
#pragma unroll
    for (int g = 0; g < T / 8; ++g) {
      const int col = 8 * g + (lt & 3) * 2;
      Ts[col * TP + row] = d[4 * g];
      Ts[(col + 1) * TP + row] = d[4 * g + 1];
      Ts[col * TP + row + 8] = d[4 * g + 2];
      Ts[(col + 1) * TP + row + 8] = d[4 * g + 3];
    }
    asm volatile("bar.sync 1, %0;" :: "n"(NC) : "memory");
    if (Z) rr_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
    constexpr int F = T * (R / 4);
    const int off = ROT ? (int)(((long long)sp * F) / S) : 0;
    for (int j = tid; j < F; j += NC) {
      int idx = j + off;
      if (idx >= F) idx -= F;
      const int tk = idx / (R / 4), qd = idx % (R / 4);
      const int c = tt * T + tk;
      if (c < M) {
        const float4 v = *reinterpret_cast<const float4*>(Ts + tk * TP + qd * 4);
        float* o = out + (size_t)c * 256 + rt * R + qd * 4;
        if (Z) asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
        else *reinterpret_cast<float4*>(o) = v;
      }
    }
  }
  if constexpr (CL > 1) rr_csync();   // no CTA exits while a peer may still arrive on its barriers
}

using RREncode = decltype(&cuTensorMapEncodeTiled);
static int rr_map(CUtensorMap* tm, const void* p, uint64_t rows, uint64_t cols, uint32_t box) {
  static RREncode encode = nullptr;
  if (!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&encode, 12000, cudaEnableDefault, &q);
    if (rc != cudaSuccess || !encode) return -100;
  }
  cuuint64_t dims[2] = {cols, rows}, strides[1] = {cols * 2};
  cuuint32_t b[2] = {64, box}, es[2] = {1, 1};
  return int(encode(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int WG, int T, int QS, int CL, int ROT>
int rr_launch_t(int64_t w, int64_t x, int64_t out, int M, int S, int64_t zc, int64_t stream, int query) {
  constexpr int R = 64 * WG, NRT = 256 / R, ST = 2 * R * 128 + T * 128, SMEM = 1024 + QS * ST + 8 * (2 * QS + 1) + 64;
  static_assert(T * (R + 4) * 4 <= QS * ST, "epilogue staging exceeds the ring");
  static_assert(SMEM <= 232448, "shared memory");
  auto fn = rr_kernel<WG, T, QS, CL, ROT>;
  static int nsm = 0;
  if (!nsm) {
    int rc = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); if (rc) return rc;
    if (CL > 8) { rc = cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); if (rc) return rc; }
    int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  }
  const int ntt = (M + T - 1) / T, grid = NRT * ntt * S;
  if (M < 1 || M > 1024 || S < 1 || S > 128 || ntt % CL) return -5;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(WG * 128 + 64); cfg.dynamicSmemBytes = SMEM; cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int resident = nsm;
  if (CL > 1) {
    int n = 0;
    int rc = cudaOccupancyMaxActiveClusters(&n, fn, &cfg); if (rc) { cudaGetLastError(); return -8; }
    resident = n * CL;
  }
  if (query) return resident;
  if (S != 1 && grid > resident) return -7;   // the in-kernel clear waits for every CTA: all must be resident
  CUtensorMap mw, mx;
  int rc = rr_map(&mw, (void*)w, 256, 16384, CL > 1 ? 2 * R / CL : R); if (rc) return rc;
  rc = rr_map(&mx, (void*)x, M, 8192, T); if (rc) return rc;
  rc = cudaLaunchKernelEx(&cfg, fn, mw, mx, (float*)out, M, S, ntt, (int*)zc);
  if (rc) return rc;
  return int(cudaGetLastError());
}
int rr_launch(int id, int64_t w, int64_t x, int64_t out, int M, int S, int64_t zc, int64_t stream, int query) {
  switch (id) {
@@CASES@@
  default: return -99;
  }
}
"""
_RR_CPP = "#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rr_launch(int,int64_t,int64_t,int64_t,int,int,int64_t,int64_t,int);\n"
# m -> (WG, T, QS, CL, ROT, S), set from the Modal screens (proj_router_v1/modal_r*.log); {} keeps the dispatcher everywhere
_RR_ROUTES = {}
_RR_LOG = []


def _rr_source(cfgs):
    fns = []
    for n in sorted({c[1] for c in cfgs}):
        regs = ", ".join("%%%d" % i for i in range(n // 2))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(n // 2))
        fns.append('template<> RD void rr_mma<%d>(float* d, uint64_t da, uint64_t db) {\n  asm volatile("wgmma.mma_async.sync.aligned.'
                   'm64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, 1, 1, 1, 0, 0;\\n" : %s : "l"(da), "l"(db));\n}'
                   % (n, n, regs, n // 2, n // 2 + 1, outs))
    kinds = sorted({c[:5] for c in cfgs})
    cases = "\n".join("  case %d: return rr_launch_t<%d, %d, %d, %d, %d>(w, x, out, M, S, zc, stream, query);" % ((i,) + k) for i, k in enumerate(kinds))
    return _RR_CUDA.replace("@@MMA@@", "\n".join(fns)).replace("@@CASES@@", cases), kinds


def _rr_state(weights, cfgs):
    import hashlib
    from torch.utils.cpp_extension import load_inline
    src, kinds = _rr_source(cfgs)
    ext = load_inline("rrouter_" + hashlib.sha256(src.encode()).hexdigest()[:12], _RR_CPP, cuda_sources=src,
                      functions=["rr_launch"], extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"],
                      no_implicit_headers=True, with_pytorch_error_handling=False)
    w = weights["router"].float()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    return {"ext": ext, "kinds": kinds, "w": torch.cat([hi, lo], dim=1).contiguous(),
            "zc": torch.zeros(4096, device=w.device, dtype=torch.int32), "slots": {}, "routes": {}}


def _rr_run(x, rst, cfg, query=0):
    """cfg = (WG, T, QS, CL, ROT, S): out [m, 256] FP32 (query=1: the co-resident CTA count instead)."""
    m = x.shape[0]
    kind, s = tuple(cfg[:5]), int(cfg[5])
    ntt = (m + kind[1] - 1) // kind[1]
    key = (kind, ntt * (256 // (64 * kind[0])) * s)   # one grid counter per (kernel, grid size)
    slot = rst["slots"].get(key)
    if slot is None:
        slot = rst["slots"][key] = len(rst["slots"])
    out = torch.empty((m, 256), device=x.device, dtype=torch.float32)
    rc = rst["ext"].rr_launch(rst["kinds"].index(kind), rst["w"].data_ptr(), x.data_ptr(), out.data_ptr(), m, s,
                              rst["zc"].data_ptr() + 4 * slot, torch.cuda.current_stream().cuda_stream, query)
    if query:
        return rc
    if rc:
        raise RuntimeError("rr launch failed: %d (cfg %s, m %d)" % (rc, cfg, m))
    return out


def _rr_inputs(m, gen, dev):
    xs = [torch.randn((m, 8192), generator=gen, device=dev)]
    xo = torch.randn((m, 8192), generator=gen, device=dev)
    xo[:, torch.randperm(8192, generator=gen, device=dev)[:81]] *= 30
    xs.append(xo)
    p = torch.zeros((m, 8192), device=dev)
    p[torch.arange(m, device=dev), torch.randint(0, 8192, (m,), generator=gen, device=dev)] = 64.0
    xs.append(p)
    return [t.to(torch.bfloat16).contiguous() for t in xs]


def _rr_valid(rst, cfg, m, w32):
    gen = torch.Generator(device="cuda").manual_seed(9100 + m)
    for x in _rr_inputs(m, gen, w32.device):
        ref = x.double() @ w32.double().t()
        for _ in range(2):   # a second launch checks the in-kernel clear and the grid counter generation
            out = _rr_run(x, rst, cfg)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(out).all()):
            return False
        d = out.double() - ref
        if float(d.norm() / ref.norm()) > 2e-5:
            return False
        rows = (d.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-30))
        if float(rows.max()) > 5e-5:
            return False
    return True


def _rr_prepare(weights):
    if not _RR_ROUTES:
        return {"routes": {}}
    cfgs = sorted(set(_RR_ROUTES.values()))
    rst = _rr_state(weights, cfgs)
    w32 = weights["router"].float()
    for m, cfg in sorted(_RR_ROUTES.items()):
        try:
            ok = _rr_valid(rst, cfg, m, w32)
        except Exception as exc:  # noqa: BLE001
            ok = False
            _RR_LOG.append("rr m=%d cfg %s failed %r" % (m, cfg, exc))
        _RR_LOG.append("rr m=%d cfg %s valid %s" % (m, cfg, ok))
        if ok:
            rst["routes"][m] = cfg
    print("\n".join(_RR_LOG), flush=True)
    return rst


_rr_parent_prepare = prepare
_rr_parent_kernel = kernel_fn


def prepare(weights):
    st = _rr_parent_prepare(weights)
    try:
        st["rr"] = _rr_prepare(weights)
    except Exception as exc:  # noqa: BLE001  (any failure leaves the dispatcher)
        print("rr router unavailable:", repr(exc)[:200], flush=True)
        st["rr"] = {"routes": {}}
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    if cfg["layer"] == "router":
        rc = state["rr"]["routes"].get(cfg["m"])
        if rc is not None:
            _ctx_bind(data["x"])
            return _rr_run(data["x"], state["rr"], rc)
    return _rr_parent_kernel(data, state)
