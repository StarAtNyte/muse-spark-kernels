#include <cuda_bf16.h>
#include <torch/extension.h>
#include <vector>

// D0.1: TPB=512, one CTA per token row, H=8192 fixed
// D0.2: C=2 chunks × 8 elements/thread = 16 elements/thread
// D0.5: __launch_bounds__(512, 2) — cap ≤ 64 regs/thread
// D0.6: float red[2][NW] — 128 B static smem

static constexpr int TPB  = 512;
static constexpr int NW   = TPB / 32;    // 16 warps
static constexpr int H    = 8192;
static constexpr int HALF = H / 2;       // 4096

__global__ __launch_bounds__(512, 2)
void boundary_kernel(
    const __nv_bfloat16* __restrict__ branch,  // [T, H] BF16
    const float*          __restrict__ stream,  // [T, H] fp32
    const __nv_bfloat16* __restrict__ gpost,   // [H] BF16  (via __ldg)
    const float*          __restrict__ alpha,   // [H] fp32  (via __ldg)
    const float*          __restrict__ beta,    // [H] fp32  (via __ldg)
    const __nv_bfloat16* __restrict__ gpre,    // [H] BF16  (via __ldg)
    float*               sout,                  // [T, H] fp32 (plain STG)
    __nv_bfloat16*       nout                   // [T, H] BF16 (plain STG)
) {
    // D0.6: 128 B static smem for two-pass cross-warp staging
    __shared__ float red[2][NW];

    const int t    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;

    // D0.8: counter-zero prologue — CTA 0 only
    // No explicit work counter in this operator's API; structural guard only.
    if (blockIdx.x == 0 && tid < NW) {
        red[0][tid] = 0.0f;
        red[1][tid] = 0.0f;
    }

    // Row base pointers
    const int row_off = t * H;
    const __nv_bfloat16* __restrict__ br = branch + row_off;
    const float*          __restrict__ st = stream + row_off;
    float*                             so = sout   + row_off;
    __nv_bfloat16*                     no = nout   + row_off;

    // D0.2: thread tid covers:
    //   chunk 0: elements [tid*8 .. tid*8+7]
    //   chunk 1: elements [HALF + tid*8 .. HALF + tid*8+7]
    const int base0 = tid * 8;
    const int base1 = HALF + tid * 8;

    // ---------------------------------------------------------------
    // Load branch (16 BF16) and stream (16 fp32) — plain LDG
    // ---------------------------------------------------------------
    float b[16], s[16];

    // branch chunk 0: 8×BF16 = 16 bytes (one LDG.E.128)
    {
        uint4 raw = *reinterpret_cast<const uint4*>(br + base0);
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
        b[0]=__bfloat162float(p[0]); b[1]=__bfloat162float(p[1]);
        b[2]=__bfloat162float(p[2]); b[3]=__bfloat162float(p[3]);
        b[4]=__bfloat162float(p[4]); b[5]=__bfloat162float(p[5]);
        b[6]=__bfloat162float(p[6]); b[7]=__bfloat162float(p[7]);
    }
    // branch chunk 1
    {
        uint4 raw = *reinterpret_cast<const uint4*>(br + base1);
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&raw);
        b[8] =__bfloat162float(p[0]); b[9] =__bfloat162float(p[1]);
        b[10]=__bfloat162float(p[2]); b[11]=__bfloat162float(p[3]);
        b[12]=__bfloat162float(p[4]); b[13]=__bfloat162float(p[5]);
        b[14]=__bfloat162float(p[6]); b[15]=__bfloat162float(p[7]);
    }
    // stream chunk 0: 8×fp32 = 32 bytes (two LDG.E.128)
    {
        float4 v0 = *reinterpret_cast<const float4*>(st + base0);
        float4 v1 = *reinterpret_cast<const float4*>(st + base0 + 4);
        s[0]=v0.x; s[1]=v0.y; s[2]=v0.z; s[3]=v0.w;
        s[4]=v1.x; s[5]=v1.y; s[6]=v1.z; s[7]=v1.w;
    }
    // stream chunk 1
    {
        float4 v0 = *reinterpret_cast<const float4*>(st + base1);
        float4 v1 = *reinterpret_cast<const float4*>(st + base1 + 4);
        s[8] =v0.x; s[9] =v0.y; s[10]=v0.z; s[11]=v0.w;
        s[12]=v1.x; s[13]=v1.y; s[14]=v1.z; s[15]=v1.w;
    }

    // ---------------------------------------------------------------
    // D0.4: load gpost via __ldg (→ LDG.E.64.CONSTANT), apply (1+gpost)
    // gate to b, accumulate ss = Σ b²
    // ---------------------------------------------------------------
    float ss = 0.0f;

    // D0.4: __ldg via uint2 (8 B = 4×BF16) → LDG.E.64.CONSTANT
    // chunk 0, first 4 elements
    {
        uint2 g = __ldg(reinterpret_cast<const uint2*>(gpost + base0));
        __nv_bfloat162 lo, hi;
        lo = *reinterpret_cast<const __nv_bfloat162*>(&g.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&g.y);
        b[0] *= 1.0f + __bfloat162float(lo.x);
        b[1] *= 1.0f + __bfloat162float(lo.y);
        b[2] *= 1.0f + __bfloat162float(hi.x);
        b[3] *= 1.0f + __bfloat162float(hi.y);
        ss += b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3];
    }
    // chunk 0, next 4 elements
    {
        uint2 g = __ldg(reinterpret_cast<const uint2*>(gpost + base0 + 4));
        __nv_bfloat162 lo, hi;
        lo = *reinterpret_cast<const __nv_bfloat162*>(&g.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&g.y);
        b[4] *= 1.0f + __bfloat162float(lo.x);
        b[5] *= 1.0f + __bfloat162float(lo.y);
        b[6] *= 1.0f + __bfloat162float(hi.x);
        b[7] *= 1.0f + __bfloat162float(hi.y);
        ss += b[4]*b[4] + b[5]*b[5] + b[6]*b[6] + b[7]*b[7];
    }
    // chunk 1, first 4 elements
    {
        uint2 g = __ldg(reinterpret_cast<const uint2*>(gpost + base1));
        __nv_bfloat162 lo, hi;
        lo = *reinterpret_cast<const __nv_bfloat162*>(&g.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&g.y);
        b[8]  *= 1.0f + __bfloat162float(lo.x);
        b[9]  *= 1.0f + __bfloat162float(lo.y);
        b[10] *= 1.0f + __bfloat162float(hi.x);
        b[11] *= 1.0f + __bfloat162float(hi.y);
        ss += b[8]*b[8] + b[9]*b[9] + b[10]*b[10] + b[11]*b[11];
    }
    // chunk 1, next 4 elements
    {
        uint2 g = __ldg(reinterpret_cast<const uint2*>(gpost + base1 + 4));
        __nv_bfloat162 lo, hi;
        lo = *reinterpret_cast<const __nv_bfloat162*>(&g.x);
        hi = *reinterpret_cast<const __nv_bfloat162*>(&g.y);
        b[12] *= 1.0f + __bfloat162float(lo.x);
        b[13] *= 1.0f + __bfloat162float(lo.y);
        b[14] *= 1.0f + __bfloat162float(hi.x);
        b[15] *= 1.0f + __bfloat162float(hi.y);
        ss += b[12]*b[12] + b[13]*b[13] + b[14]*b[14] + b[15]*b[15];
    }

    // D0.3: intra-warp SHFL.BFLY × 5 rounds → warp partial in lane 0
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffff, ss, off);

    // STS: warp leader writes to red[0]
    if (lane == 0) red[0][wid] = ss;

    // D0.3: first named BAR.SYNC id=0, 512 threads
    asm volatile("bar.sync 0, 512;" ::: "memory");

    // All threads: serial LDS.128 × 4 to sum red[0][0..15], then MUFU rsqrtf
    float r1;
    {
        const float4* rv = reinterpret_cast<const float4*>(red[0]);
        float4 v0 = rv[0], v1 = rv[1], v2 = rv[2], v3 = rv[3];
        float tot = v0.x+v0.y+v0.z+v0.w + v1.x+v1.y+v1.z+v1.w
                  + v2.x+v2.y+v2.z+v2.w + v3.x+v3.y+v3.z+v3.w;
        r1 = rsqrtf(tot * (1.0f / H) + 1e-8f);
    }

    // ---------------------------------------------------------------
    // D0.4: load alpha, beta via __ldg (→ LDG.E.128.CONSTANT)
    // Compute sout = alpha*stream + beta*(b*r1), accumulate ss2
    // D0.7: plain STG for sout stores
    // ---------------------------------------------------------------
    float sv[16];
    float ss2 = 0.0f;

    // chunk 0
    {
        float4 a0  = __ldg(reinterpret_cast<const float4*>(alpha + base0));
        float4 a1  = __ldg(reinterpret_cast<const float4*>(alpha + base0 + 4));
        float4 bt0 = __ldg(reinterpret_cast<const float4*>(beta  + base0));
        float4 bt1 = __ldg(reinterpret_cast<const float4*>(beta  + base0 + 4));

        sv[0] = a0.x*s[0]  + bt0.x*(b[0]*r1);
        sv[1] = a0.y*s[1]  + bt0.y*(b[1]*r1);
        sv[2] = a0.z*s[2]  + bt0.z*(b[2]*r1);
        sv[3] = a0.w*s[3]  + bt0.w*(b[3]*r1);
        sv[4] = a1.x*s[4]  + bt1.x*(b[4]*r1);
        sv[5] = a1.y*s[5]  + bt1.y*(b[5]*r1);
        sv[6] = a1.z*s[6]  + bt1.z*(b[6]*r1);
        sv[7] = a1.w*s[7]  + bt1.w*(b[7]*r1);

        ss2 += sv[0]*sv[0]+sv[1]*sv[1]+sv[2]*sv[2]+sv[3]*sv[3]
             + sv[4]*sv[4]+sv[5]*sv[5]+sv[6]*sv[6]+sv[7]*sv[7];

        *reinterpret_cast<float4*>(so + base0)     = make_float4(sv[0],sv[1],sv[2],sv[3]);
        *reinterpret_cast<float4*>(so + base0 + 4) = make_float4(sv[4],sv[5],sv[6],sv[7]);
    }
    // chunk 1
    {
        float4 a0  = __ldg(reinterpret_cast<const float4*>(alpha + base1));
        float4 a1  = __ldg(reinterpret_cast<const float4*>(alpha + base1 + 4));
        float4 bt0 = __ldg(reinterpret_cast<const float4*>(beta  + base1));
        float4 bt1 = __ldg(reinterpret_cast<const float4*>(beta  + base1 + 4));

        sv[8]  = a0.x*s[8]  + bt0.x*(b[8]*r1);
        sv[9]  = a0.y*s[9]  + bt0.y*(b[9]*r1);
        sv[10] = a0.z*s[10] + bt0.z*(b[10]*r1);
        sv[11] = a0.w*s[11] + bt0.w*(b[11]*r1);
        sv[12] = a1.x*s[12] + bt1.x*(b[12]*r1);
        sv[13] = a1.y*s[13] + bt1.y*(b[13]*r1);
        sv[14] = a1.z*s[14] + bt1.z*(b[14]*r1);
        sv[15] = a1.w*s[15] + bt1.w*(b[15]*r1);

        ss2 += sv[8]*sv[8]+sv[9]*sv[9]+sv[10]*sv[10]+sv[11]*sv[11]
             + sv[12]*sv[12]+sv[13]*sv[13]+sv[14]*sv[14]+sv[15]*sv[15];

        *reinterpret_cast<float4*>(so + base1)     = make_float4(sv[8], sv[9], sv[10],sv[11]);
        *reinterpret_cast<float4*>(so + base1 + 4) = make_float4(sv[12],sv[13],sv[14],sv[15]);
    }

    // D0.3: second intra-warp SHFL.BFLY × 5 rounds
    for (int off = 16; off > 0; off >>= 1)
        ss2 += __shfl_xor_sync(0xffffffff, ss2, off);

    // STS: warp leader writes to red[1]
    if (lane == 0) red[1][wid] = ss2;

    // D0.3: second named BAR.SYNC id=0, 512 threads
    asm volatile("bar.sync 0, 512;" ::: "memory");

    // All threads: serial LDS.128 × 4 to sum red[1][0..15], then MUFU rsqrtf
    float r2;
    {
        const float4* rv = reinterpret_cast<const float4*>(red[1]);
        float4 v0 = rv[0], v1 = rv[1], v2 = rv[2], v3 = rv[3];
        float tot2 = v0.x+v0.y+v0.z+v0.w + v1.x+v1.y+v1.z+v1.w
                   + v2.x+v2.y+v2.z+v2.w + v3.x+v3.y+v3.z+v3.w;
        r2 = rsqrtf(tot2 * (1.0f / H) + 1e-5f);
    }

    // ---------------------------------------------------------------
    // D0.4: load gpre via __ldg (→ LDG.E.64.CONSTANT)
    // nout = bf16(sout * r2 * gpre)
    // D0.7: plain STG for nout stores
    // ---------------------------------------------------------------

    // D0.4: __ldg via uint2 (8 B = 4×BF16) → LDG.E.64.CONSTANT
    // Pass 2: nout = bf16(sout * r2 * gpre)  — no (1+) for gpre
    // chunk 0
    {
        uint2 g0 = __ldg(reinterpret_cast<const uint2*>(gpre + base0));
        uint2 g1 = __ldg(reinterpret_cast<const uint2*>(gpre + base0 + 4));
        __nv_bfloat162 lo0, hi0, lo1, hi1;
        lo0 = *reinterpret_cast<const __nv_bfloat162*>(&g0.x);
        hi0 = *reinterpret_cast<const __nv_bfloat162*>(&g0.y);
        lo1 = *reinterpret_cast<const __nv_bfloat162*>(&g1.x);
        hi1 = *reinterpret_cast<const __nv_bfloat162*>(&g1.y);

        __nv_bfloat162 n01 = __floats2bfloat162_rn(
            sv[0]*r2*__bfloat162float(lo0.x),
            sv[1]*r2*__bfloat162float(lo0.y));
        __nv_bfloat162 n23 = __floats2bfloat162_rn(
            sv[2]*r2*__bfloat162float(hi0.x),
            sv[3]*r2*__bfloat162float(hi0.y));
        __nv_bfloat162 n45 = __floats2bfloat162_rn(
            sv[4]*r2*__bfloat162float(lo1.x),
            sv[5]*r2*__bfloat162float(lo1.y));
        __nv_bfloat162 n67 = __floats2bfloat162_rn(
            sv[6]*r2*__bfloat162float(hi1.x),
            sv[7]*r2*__bfloat162float(hi1.y));

        uint4 pack;
        unsigned* pp = reinterpret_cast<unsigned*>(&pack);
        pp[0] = *reinterpret_cast<const unsigned*>(&n01);
        pp[1] = *reinterpret_cast<const unsigned*>(&n23);
        pp[2] = *reinterpret_cast<const unsigned*>(&n45);
        pp[3] = *reinterpret_cast<const unsigned*>(&n67);
        *reinterpret_cast<uint4*>(no + base0) = pack;
    }
    // chunk 1
    {
        uint2 g0 = __ldg(reinterpret_cast<const uint2*>(gpre + base1));
        uint2 g1 = __ldg(reinterpret_cast<const uint2*>(gpre + base1 + 4));
        __nv_bfloat162 lo0, hi0, lo1, hi1;
        lo0 = *reinterpret_cast<const __nv_bfloat162*>(&g0.x);
        hi0 = *reinterpret_cast<const __nv_bfloat162*>(&g0.y);
        lo1 = *reinterpret_cast<const __nv_bfloat162*>(&g1.x);
        hi1 = *reinterpret_cast<const __nv_bfloat162*>(&g1.y);

        __nv_bfloat162 n01 = __floats2bfloat162_rn(
            sv[8] *r2*__bfloat162float(lo0.x),
            sv[9] *r2*__bfloat162float(lo0.y));
        __nv_bfloat162 n23 = __floats2bfloat162_rn(
            sv[10]*r2*__bfloat162float(hi0.x),
            sv[11]*r2*__bfloat162float(hi0.y));
        __nv_bfloat162 n45 = __floats2bfloat162_rn(
            sv[12]*r2*__bfloat162float(lo1.x),
            sv[13]*r2*__bfloat162float(lo1.y));
        __nv_bfloat162 n67 = __floats2bfloat162_rn(
            sv[14]*r2*__bfloat162float(hi1.x),
            sv[15]*r2*__bfloat162float(hi1.y));

        uint4 pack;
        unsigned* pp = reinterpret_cast<unsigned*>(&pack);
        pp[0] = *reinterpret_cast<const unsigned*>(&n01);
        pp[1] = *reinterpret_cast<const unsigned*>(&n23);
        pp[2] = *reinterpret_cast<const unsigned*>(&n45);
        pp[3] = *reinterpret_cast<const unsigned*>(&n67);
        *reinterpret_cast<uint4*>(no + base1) = pack;
    }
}

std::vector<torch::Tensor> run(
    torch::Tensor branch, torch::Tensor stream,
    torch::Tensor gpost,  torch::Tensor alpha,
    torch::Tensor beta,   torch::Tensor gpre)
{
    const int T = static_cast<int>(branch.size(0));
    TORCH_CHECK(branch.size(1) == H, "H must be 8192");
    TORCH_CHECK(branch.is_cuda() && stream.is_cuda(), "inputs must be CUDA tensors");

    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(branch.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(branch.device());
    auto sout_t = torch::empty({T, H}, opts_f32);
    auto nout_t = torch::empty({T, H}, opts_bf16);

    boundary_kernel<<<T, TPB>>>(
        reinterpret_cast<const __nv_bfloat16*>(branch.data_ptr()),
        stream.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(gpost.data_ptr()),
        alpha.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr()),
        sout_t.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(nout_t.data_ptr())
    );

    return {sout_t, nout_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
