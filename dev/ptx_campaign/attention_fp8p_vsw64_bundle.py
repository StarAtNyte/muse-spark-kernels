# okc: problem=muse_attention_decode
"""Fuse QK normalization, RoPE, query conversion, and paged cache writes.

FA3 supplies the exact all-key attention path. No cache content or activation
is retained across calls. The only FP8 rounding is that allowed by the task.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


@triton.jit
def _prepare_qkv(X, POS, LOC, KS, VS, Q, KC, VC,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 STRIDE: tl.constexpr, THETA: tl.constexpr,
                 SCALE: tl.constexpr, HEADS: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0)
    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    dims = tl.arange(0, D)
    valid = heads < HQ + HKV
    values = tl.load(X + token * STRIDE + heads[:, None] * D + dims[None, :],
                     valid[:, None], 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, 1) / D + 1.e-5)
    normalized = values * inv_rms[:, None]
    if THETA != 0:
        partner = tl.load(X + token * STRIDE + heads[:, None] * D + (dims[None, :] ^ 1),
                          valid[:, None], 0).to(tl.float32) * inv_rms[:, None]
        position = tl.load(POS + token).to(tl.float32)
        frequency = tl.exp((-2.0 * (dims // 2).to(tl.float32) / D) * tl.log(THETA))
        angle = position * frequency
        cs, sn = libdevice.cos(angle), libdevice.sin(angle)
        normalized = normalized * cs[None, :] + tl.where(
            (dims[None, :] & 1) == 0, -partner, partner) * sn[None, :]
    # Query factor 5.473 is folded into FA3's softmax scale, matching SGLang.
    tl.store(Q + token * HQ * D + heads[:, None] * D + dims[None, :],
             normalized, (heads < HQ)[:, None])
    local_head = heads - HQ
    is_key = (heads >= HQ) & valid
    slot = tl.load(LOC + token)
    if SCALE:
        key_scale = tl.load(KS)
        value_scale = tl.load(VS)
    else:
        key_scale, value_scale = 1., 1.
    tl.store(KC + slot * HKV * D + local_head[:, None] * D + dims[None, :],
             normalized / key_scale, is_key[:, None])
    v = tl.load(X + token * STRIDE + (HQ + HKV + local_head[:, None]) * D + dims[None, :],
                is_key[:, None], 0).to(tl.float32)
    tl.store(VC + slot * HKV * D + local_head[:, None] * D + dims[None, :],
             v / value_scale, is_key[:, None])


@triton.jit
def _sliding_partials(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE,
                      HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                      LENGTH: tl.constexpr, PAGE: tl.constexpr, TABLE_N: tl.constexpr,
                      WINDOW: tl.constexpr, SPLITS: tl.constexpr,
                      CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SM_SCALE: tl.constexpr, SCALED: tl.constexpr):
    b, kh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    GROUP: tl.constexpr = HQ // HKV
    rows = tl.arange(0, BM)
    dims = tl.arange(0, D)
    token = rows // GROUP
    qh = kh * GROUP + rows % GROUP
    prefix = tl.load(SEQLEN + b)
    position = prefix + token
    if WINDOW:
        first = tl.maximum(prefix - WINDOW + 1, 0)
    else:
        first = 0
    key_scale, value_scale = 1., 1.
    if SCALED:
        key_scale, value_scale = tl.load(KS), tl.load(VS)
    q = tl.load(Q + ((b * LENGTH + token[:, None]) * HQ + qh[:, None]) * D + dims[None, :],
                (rows < LENGTH * GROUP)[:, None], 0.0)
    max_score = tl.full((BM,), -float('inf'), tl.float32)
    normalizer = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    begin = first + split * CHUNK
    end = tl.minimum(begin + CHUNK, prefix + LENGTH)
    for start in range(begin, end, BN):
        logical = start + tl.arange(0, BN)
        pages = logical // PAGE
        valid = (logical < prefix + LENGTH) & (pages < TABLE_N)
        physical = tl.load(TABLE + b * TABLE_N + pages, valid, -1)
        valid = valid & (physical >= 0)
        cache_row = tl.maximum(physical, 0) * PAGE + logical % PAGE
        k = tl.load(KC + (cache_row[None, :] * HKV + kh) * D + dims[:, None],
                    valid[None, :], 0.0)
        scores = tl.dot(q, k) * (SM_SCALE * key_scale)
        visible = (valid[None, :] & (logical[None, :] <= position[:, None]) &
                   ((WINDOW == 0) | (logical[None, :] > position[:, None] - WINDOW)) &
                   (rows < LENGTH * GROUP)[:, None])
        scores = tl.where(visible, scores, -float('inf'))
        next_max = tl.maximum(max_score, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        correction = tl.exp(max_score - safe_max)
        probabilities = tl.exp(scores - safe_max[:, None])
        normalizer = normalizer * correction + tl.sum(probabilities, 1)
        acc *= correction[:, None]
        v = tl.load(VC + (cache_row[:, None] * HKV + kh) * D + dims[None, :],
                    valid[:, None], 0.0).to(tl.bfloat16)
        acc = tl.dot(probabilities.to(tl.bfloat16), v, acc)
        max_score = next_max
    inverse = 1. / tl.where(normalizer > 0, normalizer, 1.)
    output = acc * inverse[:, None] * value_scale
    logsumexp = tl.where(normalizer > 0, max_score + tl.log(normalizer), -float('inf'))
    base = ((b * HKV + kh) * SPLITS + split) * BM
    tl.store(PART + (base + rows[:, None]) * D + dims[None, :], output)
    tl.store(LSE + base + rows, logsumexp)


@triton.jit
def _merge_norm_gate(PART, LSE, X, OUT,
                     HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                     LENGTH: tl.constexpr, STRIDE: tl.constexpr,
                     SPLITS: tl.constexpr, BM: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    t, qh = tl.program_id(0), tl.program_id(1)
    b, token = t // LENGTH, t % LENGTH
    GROUP: tl.constexpr = HQ // HKV
    kh, row = qh // GROUP, token * GROUP + qh % GROUP
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    bases = ((b * HKV + kh) * SPLITS + splits) * BM + row
    logs = tl.load(LSE + bases)
    high = tl.max(logs, 0)
    factors = tl.exp(logs - high)
    factors /= tl.sum(factors, 0)
    partial = tl.load(PART + bases[:, None] * D + dims[None, :])
    value = tl.sum(partial * factors[:, None], 0).to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / D + 1.e-5)
    gate = tl.load(X + t * STRIDE + (HQ + 2 * HKV + qh) * D + dims).to(tl.float32)
    output = value * inv_rms * tl.sigmoid(gate)
    tl.store(OUT + (t * HQ + qh) * D + dims, output)


def _v7_kernel_fn(data):
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
    # Custom decode needs enough independent CTAs at larger batches.
    # Verify blocks use FA3 for long contexts or larger request batches.
    context_bound = data['page_table'].shape[1] * cfg['page_size']
    use_sliding = cfg['window'] and (batch <= 4 or (length == 1 and batch*hkv*4 >= 192))
    use_full = not cfg['window'] and (length == 1 or (batch <= 4 and context_bound <= 131072))
    if use_sliding or use_full:
        # Partition the actual window, never the entire logical context.
        # The extra LENGTH-1 rows cover the union of the causal verify windows.
        if cfg['window']:
            splits = 16 if batch <= 4 else 4
            block_n = 64
            span = cfg['window'] + length - 1
        else:
            splits = min(128, max(4, triton.next_power_of_2(triton.cdiv(512, batch * hkv))))
            block_n = 128
            span = data['page_table'].shape[1] * cfg['page_size']
        chunk = triton.cdiv(triton.cdiv(span, splits), block_n) * block_n
        bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
        partial = torch.empty((batch, hkv, splits, bm, d), device=qkv.device, dtype=torch.float32)
        lse = torch.empty((batch, hkv, splits, bm), device=qkv.device, dtype=torch.float32)
        result = torch.empty((batch, length, hq*d), device=qkv.device, dtype=torch.bfloat16)
        _sliding_partials[(batch, hkv, splits)](
            q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
            data['k_scale'], data['v_scale'], partial, lse, hq, hkv, d,
            length, cfg['page_size'], data['page_table'].shape[1], cfg['window'],
            splits, chunk, bm, block_n, cfg['sm_scale'] * cfg['qk_scale_factor'],
            scaled, num_warps=4, num_stages=2)
        _merge_norm_gate[(tokens, hq)](
            partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0),
            splits, bm, num_warps=4)
        return result
    descale = {}
    if scaled:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv),
                       v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    window = cfg['window']
    out = flash_attn_with_kvcache(
        q=q, k_cache=data['k_cache'], v_cache=data['v_cache'],
        page_table=data['page_table'], cache_seqlens=data['cache_seqlens'],
        cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=length,
        softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'],
        causal=True, window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data['scheduler_metadata'], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2*hkv)*d:]
    result = rms_norm_gate(out.reshape(tokens, hq*d).to(torch.bfloat16), gate, None,
                           head_dim=d, eps=1.e-5, has_norm=True)
    return result.view(batch, length, hq*d)


# ---- CUDA split-KV flash decode ----
_WG_CUDA = '\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define HD 64\n#define WT 64              // keys per tile\n\n__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ uint32_t pack_h2(float a, float b) {\n  __half2 h = __floats2half2_rn(a, b);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;\n}\n__device__ __forceinline__ uint32_t cvt_fp8x2(uint32_t x) {   // low two e4m3 bytes -> f16x2\n  uint32_t r;\n  asm("{\\n.reg .b16 a, b;\\nmov.b32 {a, b}, %1;\\ncvt.rn.f16x2.e4m3x2 %0, a;\\n}" : "=r"(r) : "r"(x));\n  return r;\n}\n\n__device__ __forceinline__ void wg_qk(float* d, uint64_t da, uint64_t db, int accum) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"\n               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "l"(da), "l"(db), "r"(accum));\n#endif\n}\n\n__device__ __forceinline__ void wg_pv(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n\n__device__ __forceinline__ float ex2f(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ void wg_pv72(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n__device__ __forceinline__ void wg_pvt(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n\n\n__device__ __forceinline__ void wg_commit() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_wait0() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum);\n__device__ __forceinline__ void wg_qk128(float* d, uint64_t da, uint64_t db, int accum) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %66, 0;\\n"\n               "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n               : "l"(da), "l"(db), "r"(accum));\n#endif\n}\n__device__ __forceinline__ uint64_t mkdesc(const void* p, uint32_t lbo, uint32_t sbo) {\n  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);\n  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_commit_wait() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory"); }\n\n// core-matrix layout of a [rows][64 byte] fp8 tile: 8 rows x 16 B cores, K cores 128 B apart, row groups 512 B apart\n__device__ __forceinline__ int core_off(int r, int chunk) { return (r >> 3) * 512 + chunk * 128 + (r & 7) * 16; }\n// vT: [d][key] f16, cores of 8 d-rows x 8 keys (16 B per row): key cores 128 B apart, d groups 1024 B apart\n__device__ __forceinline__ int vt_off(int d, int key) { return (d >> 3) * 1024 + (key >> 3) * 128 + (d & 7) * 16 + (key & 7) * 2; }\n\n__device__ __forceinline__ int vmn_off(int key, int d) { return (key >> 3) * 1152 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2; }\n__device__ __forceinline__ void pv8_ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void wg_pv8(float* c,uint32_t* a,uint64_t b){\n asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,1,1,1;"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));\n}\n\n__device__ __forceinline__ uint32_t packp8(float a,float b){uint16_t y;\n asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(y):"f"(a),"f"(b));return y;}\n\n__device__ __forceinline__ int v64_off(int row,int chunk){int a=row*64+chunk*16;return a^((a>>3)&0x30);}\n#define WS 2   // raw stages\n#define TILEB 8192   // bytes of a raw K or V tile\n__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum) { wg_qk128(d, da, db, accum); }\n\n\n__device__ __forceinline__ void wg_qk_first(float* d, uint64_t da, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 0, 1, 1;\\n"\n               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])\n               : "l"(da), "l"(db));\n#endif\n}\n__device__ __forceinline__ void wg_qk_acc(float* d, uint64_t da, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1;\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "l"(da), "l"(db));\n#endif\n}\n\n#define LAZY_TH 0.0f\n__global__ void __launch_bounds__(128, 4)\nwg5_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,\n             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,\n             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ,\n             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  extern __shared__ __align__(1024) uint8_t smem_raw[];\n  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);\n  const int tid = threadIdx.x, wgi = 0, tw = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;\n  const int g = lane >> 2, t = lane & 3;\n  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;\n  const int prefix = SEQLEN[b];\n  const int first = window ? max(prefix - window + 1, 0) : 0;\n  const int begin = first + split * chunk;\n  const int end = min(begin + chunk, prefix + L);\n  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;\n  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;\n  const long base_row = ((long)(b * 2 + kh) * splits + split) * 64;\n  const int n = end > begin ? end - begin : 0;\n  const int nst = (n + WT - 1) / WT;\n\n  uint8_t* qs = smem;\n  uint8_t* vt = smem + 4096;\n  uint8_t* stages = smem + 4096 + 4096;\n  int* rows = reinterpret_cast<int*>(stages + WS * 8192);\n\n  const int pshift = __ffs(page) - 1, pmask = page - 1;\n  for (int i = tid; i < n; i += 128) {\n    const int tok = begin + i;\n    const int pg = tok >> pshift;\n    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;\n    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));\n  }\n  for (int c = tw; c < 64 * 4; c += 128) {\n    const int r = c >> 2, ch = c & 3;\n    const int token = r >> 3, qh = kh * 8 + (r & 7);\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (token < L) v = *reinterpret_cast<const uint4*>(Q + ((long)(b * L + token) * HQ + qh) * HD + ch * 16);\n    *reinterpret_cast<uint4*>(qs + core_off(r, ch)) = v;\n  }\n  proxy_fence();\n  __syncthreads();\n\n  // ---- loop-invariant addressing ----\n  const uint32_t stages_s = (uint32_t)__cvta_generic_to_shared(stages);\n  const int ci = tid & 7;                   // 16-byte chunk within the key row (K 0-7, V 8-15)\n  const int cc = ci & 3;\n  const bool isv = ci >= 4;\n  const uint8_t* srcb = (isv ? VC : KC) + kh * 64 + cc * 16;\n  uint32_t dst_off[4];\n  int key4[4];\n#pragma unroll\n  for (int i = 0; i < 4; i++) {\n    const int key = (tid >> 3) + 16 * i;\n    key4[i] = key;\n    dst_off[i] = isv ? (4096 + key * 64 + (cc ^ (key & 3)) * 16) : core_off(key, cc);\n  }\n  auto issue = [&](int s, int stage) {\n    if (s < nst && !(dbg & 8)) {\n      const uint32_t base = stages_s + stage * 8192;\n#pragma unroll\n      for (int i = 0; i < 4; i++) {\n        const int li = s * WT + key4[i];\n        const long rr = li < n ? rows[li] : 0;\n        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"(base + dst_off[i]), "l"(srcb + rr * 128));\n      }\n    }\n    cp_commit();\n  };\n  int wr_stage = 0;\n#pragma unroll\n  for (int s = 0; s < WS - 1; s++) { issue(s, wr_stage); wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1; }\n\n  const uint64_t qd0 = mkdesc(qs, 128, 512), qd1 = mkdesc(qs + 256, 128, 512);\n  const uint64_t kd0 = mkdesc(stages, 128, 512);\n  const uint64_t vd0 = mkdesc(vt,16,512)|(2ull<<62);\n  // per-thread V conversion offsets (loop invariant)\n  // rotated chunk assignment: lanes take consecutive keys but different 16 B chunks, so both the smem loads and the stores are (nearly) conflict free\n  const int vkey = (tw & 31) + 32 * (tw >> 6);\n  const int vcb = ((tw >> 5) & 1) * 2;\n  uint32_t vsrc_off[2], vdst_off[4];\n#pragma unroll\n  for (int j = 0; j < 2; j++) {\n    const int cs = vcb + j;\n    vsrc_off[j] = vkey * 64 + (cs ^ (vkey & 3)) * 16;\n    vdst_off[2 * j] = (uint32_t)vmn_off(vkey, 16 * cs);\n    vdst_off[2 * j + 1] = (uint32_t)vmn_off(vkey, 16 * cs + 8);\n  }\n\n  float o[32];\n  float ls_run[2]={0.f,0.f};\n#pragma unroll\n  for (int i = 0; i < 32; i++) o[i] = 0.f;\n  float m_run[2] = {-INFINITY, -INFINITY};\n\n  int rd_stage = 0;\n#pragma unroll 1\n  for (int s = 0; s < nst; s++) {\n    cp_wait<WS - 2>();\n    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }\n    issue(s + WS - 1, wr_stage);\n    wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1;\n    const uint8_t* vst = stages + rd_stage * 8192 + 4096;\n    const uint64_t kdesc = kd0 + (uint64_t)(rd_stage * 512);\n    rd_stage = rd_stage + 1 == WS ? 0 : rd_stage + 1;\n\n    float sacc[32];\n    // Key-permuted FP8 transpose: P uses the identical within-block order.\n#pragma unroll\n    for(int pr=0;pr<2;pr++){\n      int key0=warp*16+(lane&7)+8*((lane>>3)&1),byte=pr*32+16*(lane>>4);\n      uint32_t r[4];pv8_ldt(r,vst+key0*64+(((byte>>4)^(key0&3))<<4));\n      uint32_t z[4]={__byte_perm(r[0],r[1],0x6420),__byte_perm(r[0],r[1],0x7531),\n                     __byte_perm(r[2],r[3],0x6420),__byte_perm(r[2],r[3],0x7531)};\n      int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);\n      uint32_t ad=__cvta_generic_to_shared(vt+v64_off(d,warp));\n      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");\n    }\n    proxy_fence();\n    wg_fence();\n    wg_qk_first(sacc, qd0, kdesc);\n    wg_qk_acc(sacc, qd1, kdesc + 16);\n    wg_commit();\n    wg_wait0();\n#pragma unroll\n    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(sacc[i])::"memory");\n    asm volatile("bar.sync %0, 128;\\n" ::"r"(1 + wgi) : "memory");\n\n    const int tb = begin + s * WT;\n    const bool full_vis = (tb + WT <= end) && (L == 1 || tb + WT - 1 <= prefix) && (!window || L == 1 || tb > prefix + L - 1 - window);\n    if (!full_vis) {\n#pragma unroll\n      for (int j = 0; j < 8; j++)\n#pragma unroll\n        for (int r = 0; r < 4; r++) {\n          const int row = warp * 16 + g + 8 * (r >> 1);\n          const int token = row >> 3;\n          const int col = tb + j * 8 + 2 * t + (r & 1);\n          const int pos = prefix + token;\n          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);\n          if (!vis) sacc[4 * j + r] = -INFINITY;\n        }\n    }\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int j = 0; j < ((dbg & 64) ? 0 : 8); j++) {\n      mx0 = fmaxf(mx0, fmaxf(sacc[4 * j], sacc[4 * j + 1]));\n      mx1 = fmaxf(mx1, fmaxf(sacc[4 * j + 2], sacc[4 * j + 3]));\n    }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));\n    mx0 *= sc2; mx1 *= sc2;\n    // lazy rescale: the reference max only moves when a tile exceeds it by more than LAZY_TH (log2 domain), so p <= 2^LAZY_TH\n    if (__any_sync(0xffffffffu, mx0 > m_run[0] + LAZY_TH || mx1 > m_run[1] + LAZY_TH)) {\n      const float n0 = fmaxf(m_run[0], mx0), n1 = fmaxf(m_run[1], mx1);\n      const float s0 = n0 == -INFINITY ? 0.f : n0, s1 = n1 == -INFINITY ? 0.f : n1;\n      const float c0 = ex2f(m_run[0] - s0), c1 = ex2f(m_run[1] - s1);\n      m_run[0] = n0; m_run[1] = n1;ls_run[0]*=c0;ls_run[1]*=c1;\n#pragma unroll\n      for (int j = 0; j < 8; j++) {\n        o[4 * j] *= c0; o[4 * j + 1] *= c0;\n        o[4 * j + 2] *= c1; o[4 * j + 3] *= c1;\n      }\n    }\n    const float sf0 = m_run[0] == -INFINITY ? 0.f : m_run[0], sf1 = m_run[1] == -INFINITY ? 0.f : m_run[1];\n    uint32_t pa[8][2];\n#pragma unroll\n    for (int j = 0; j < 8; j++) {\n      const float p0 = (dbg & 32) ? sacc[4 * j] : ex2f(fmaf(sacc[4 * j], sc2, 8.f-sf0)), p1 = (dbg & 32) ? sacc[4 * j + 1] : ex2f(fmaf(sacc[4 * j + 1], sc2, 8.f-sf0));\n      const float p2 = (dbg & 32) ? sacc[4 * j + 2] : ex2f(fmaf(sacc[4 * j + 2], sc2, 8.f-sf1)), p3 = (dbg & 32) ? sacc[4 * j + 3] : ex2f(fmaf(sacc[4 * j + 3], sc2, 8.f-sf1));\n      ls_run[0]+=p0+p1;ls_run[1]+=p2+p3;\n      pa[j][0] = packp8(p0,p1);\n      pa[j][1] = packp8(p2,p3);\n    }\n    uint32_t pp[2][4];\n#pragma unroll\n    for(int kk=0;kk<2;kk++){\n#pragma unroll\n      for(int h=0;h<2;h++){\n        pp[kk][h]=pa[kk*4][h]|(pa[kk*4+1][h]<<16);\n        pp[kk][h+2]=pa[kk*4+2][h]|(pa[kk*4+3][h]<<16);\n      }\n    }\n    wg_fence();wg_pv8(o,pp[0],vd0);wg_pv8(o,pp[1],vd0+2);\n    wg_commit_wait();\n#pragma unroll\n    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(o[i])::"memory");\n  }\n  cp_wait<0>();\n#pragma unroll\n  for (int h = 0; h < 2; h++) {\n    const int row = warp * 16 + g + 8 * h;\n    float ls=ls_run[h];ls+=__shfl_xor_sync(0xffffffff,ls,1);ls+=__shfl_xor_sync(0xffffffff,ls,2);\n    const float inv = ls > 0.f ? 1.f / ls : 1.f;\n#pragma unroll\n    for (int j = 0; j < 8; j++) {\n      PART[(base_row + row) * HD + j * 8 + 2 * t] = o[4 * j + 2 * h] * inv * value_scale;\n      PART[(base_row + row) * HD + j * 8 + 2 * t + 1] = o[4 * j + 2 * h + 1] * inv * value_scale;\n    }\n    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls) - 8.f) * 0.6931471805599453f : -INFINITY;\n  }\n#endif\n}\n\nextern "C" int wg5_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,\n                       const float* VS, float* PART, float* LSE, int B, int L, int HQ, int table_n, int page,\n                       int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {\n  const size_t rmax = ((chunk + 255) / 256) * 256;\n  const size_t dyn = 1024 + 4096 + 4096 + (size_t)WS * 8192 + rmax * 4;\n  cudaFuncSetAttribute(wg5_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);\n  dim3 grid(splits, 2, B);\n  wg5_partials<<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),\n                                       reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,\n                                       table_n, page, window, splits, chunk, sm_scale, scaled, dbg);\n  return 0;\n}\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nextern "C" int wg5_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,\n                       int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);\nvoid wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,\n               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,\n               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {\n  int r = wg5_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),\n                  vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,\n                  (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,\n                  at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(r == 0, "wg5");\n}\n'
_WG_CPP = r'''void wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg);'''

_WG_MOD = None
_WG_ON = True
_WG_MINB = 8
_WG_SPLITS = 16


def _wg_module():
    global _WG_MOD
    if _WG_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WG_MOD = load_inline("muse_wg5_fp8p_vsw64", _WG_CPP, cuda_sources=_WG_CUDA, functions=["wg5_run_t"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _WG_MOD


def _wg_forward(data, q, qkv, splits):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    span = data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 64) * 64
    partial = torch.empty((batch, hkv, splits, 64, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, 64), device=qkv.device, dtype=torch.float32)
    _wg_module().wg5_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                           data['v_scale'], partial, lse, length, hq, cfg['page_size'], 0, splits, chunk,
                           cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), 0)
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)
    return result

_FD_CUDA = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
static int g_fd_pdl = 0;
extern "C" void fd_set_pdl(int v) { g_fd_pdl = v; }
template <typename... KArgs, typename... Args>
static void launch_pdl(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_fd_pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

#define HD 64
#define STAGE 16          // tokens per warp stage
#define KP 144            // bytes per converted f16 row (64 halves + 16 pad)
#define RP 80             // bytes per raw fp8 row slot (64 + 16 pad)

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t hmul2_h(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// four e4m3 bytes -> two f16x2 (exact)
__device__ __forceinline__ void fp8x4_to_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
#if defined(FD_HW_CVT) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi) : "r"(w));
  return;
#endif
  uint32_t tl = prmt(w, 0u, 0x1404u), th = prmt(w, 0u, 0x3424u);
  tl = (tl & 0x80008000u) | ((tl >> 1) & 0x3F803F80u);
  th = (th & 0x80008000u) | ((th >> 1) & 0x3F803F80u);
  lo = hmul2_h(tl, 0x5C005C00u);  // x 2^8 undoes the exponent-bias shift
  hi = hmul2_h(th, 0x5C005C00u);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((uint32_t)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)),
               "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// grid (splits, HKV, B), 128 threads.  Each warp owns every 4th 16-token stage of the CTA's range.
template <int MT, int S, bool QK8>
__global__ void __launch_bounds__(128)
fd_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
            const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
            const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
            int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = MT * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;  // scores in log2 domain
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;

  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  // ---- per-warp shared layout ----
  constexpr int RAW = 2 * 16 * RP;                       // K and V raw slot
  constexpr int PER_WARP = S * RAW + 2 * 16 * KP;        // ring + converted K and V
  uint8_t* wbase = smem + warp * PER_WARP;
  uint8_t* kf = wbase + S * RAW;
  uint8_t* vf = kf + 16 * KP;

  // ---- Q fragments (f16, or raw e4m3 for the fp8 mma path) ----
  uint32_t qa[MT][4][4];
  if constexpr (QK8) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int row = mt * 16 + g + 8 * half;
        const int token = row >> 3, qh = kh * 8 + (row & 7);
        const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          uint32_t x0 = 0, x1 = 0;
          if (token < L) {
            x0 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
            x1 = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
          }
          qa[mt][ks][half ? 1 : 0] = x0;  // a0 / a1
          qa[mt][ks][half ? 3 : 2] = x1;  // a2 / a3
        }
      }
    }
  } else {
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = mt * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const bool ok = token < L;
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        // elements d = 16 ks + 2t,2t+1 (a0/a1) and 16 ks + 8 + 2t,2t+1 (a2/a3)
        uint32_t lo = 0, hi = 0, x = 0, y = 0;
        if (ok) {
          uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
          uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
          fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
          x = lo;
          y = hi;
        }
        qa[mt][ks][half ? 1 : 0] = x;       // a0 (row g) / a1 (row g+8): first column pair
        qa[mt][ks][half ? 3 : 2] = y;       // a2 / a3: second column pair
      }
    }
  }
  }

  float o[MT][8][4];
  float m_run[MT][2], l_run[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++)
#pragma unroll
      for (int r = 0; r < 4; r++) o[mt][d][r] = 0.f;
  }

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  // stage s of this warp covers tokens begin + (warp + 4 s) * STAGE ...
  auto issue = [&](int s, long r0, long r1) {
    if (s < (nstages - warp + 3) / 4) {
      uint8_t* slot = wbase + (s % S) * RAW;
      const int c = lane & 3;
      const int tok0 = lane >> 2;
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const long r = i ? r1 : r0;
        const int tok = tok0 + 8 * i;
        // invalid rows read row 0 (masked later)
        const long rr = r < 0 ? 0 : r;
        cp16(slot + tok * RP + c * 16, KC + (rr * HKV + kh) * HD + c * 16);
        cp16(slot + 16 * RP + tok * RP + c * 16, VC + (rr * HKV + kh) * HD + c * 16);
      }
    }
    cp_commit();
  };

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto stage_tok = [&](int s) { return begin + (warp + 4 * s) * STAGE; };
  // gather-index prefetch for stage s: tokens stage_tok(s) + lane/4 and +8
  long ra0 = -1, ra1 = -1, rb0 = -1, rb1 = -1;
  auto load_rows = [&](int s, long& x0, long& x1) {
    const int tb = stage_tok(s) + (lane >> 2);
    x0 = x1 = -1;
    if (s < my_stages) {
      if (tb < end) x0 = row_of(tb);
      if (tb + 8 < end) x1 = row_of(tb + 8);
    }
  };
  {
    long pr0[S], pr1[S];
#pragma unroll
    for (int s = 0; s < S; s++) load_rows(s, pr0[s], pr1[s]);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr0[s], pr1[s]);
    rb0 = pr0[S - 1];
    rb1 = pr1[S - 1];
  }

  const int srow = lane & 7;  // ldmatrix row within an 8x8 tile
  const int smat = lane >> 3; // matrix index within x4

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1, rb0, rb1);
    load_rows(s + S, rb0, rb1);
    const uint8_t* slot = wbase + (s % S) * RAW;
    // ---- convert K and V rows of this stage to f16 (lane handles one 16B raw chunk x 2) ----
    {
      const int tk = lane >> 1, half16 = (lane & 1) * 32;  // 16 tokens x 2 halves of 32 bytes
#pragma unroll
      for (int which = (QK8 ? 1 : 0); which < 2; which++) {
        const uint8_t* src = slot + which * 16 * RP + tk * RP + half16;
        uint8_t* dst = (which ? vf : kf) + tk * KP + half16 * 2;
        const uint4 u0 = *reinterpret_cast<const uint4*>(src);
        const uint4 u1 = *reinterpret_cast<const uint4*>(src + 16);
        uint4 o0, o1, o2, o3;
        fp8x4_to_h2(u0.x, o0.x, o0.y);
        fp8x4_to_h2(u0.y, o0.z, o0.w);
        fp8x4_to_h2(u0.z, o1.x, o1.y);
        fp8x4_to_h2(u0.w, o1.z, o1.w);
        fp8x4_to_h2(u1.x, o2.x, o2.y);
        fp8x4_to_h2(u1.y, o2.z, o2.w);
        fp8x4_to_h2(u1.z, o3.x, o3.y);
        fp8x4_to_h2(u1.w, o3.z, o3.w);
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
        *reinterpret_cast<uint4*>(dst + 32) = o2;
        *reinterpret_cast<uint4*>(dst + 48) = o3;
      }
    }
    __syncwarp();

    const int tb = stage_tok(s);
    const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                          (!window || L == 1 || tb > prefix + L - 1 - window);
    // ---- S = Q K^T for each m-tile: 2 n-tiles of 8 tokens ----
    uint32_t kb[2][4][2];
    if constexpr (QK8) {
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          const uint8_t* kr = slot + (nt * 8 + g) * RP + 32 * ks + 4 * t;
          kb[nt][ks][0] = *reinterpret_cast<const uint32_t*>(kr);
          kb[nt][ks][1] = *reinterpret_cast<const uint32_t*>(kr + 16);
        }
    } else {
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4(r0, r1, r2, r3, kf + (nt * 8 + srow) * KP + pr * 64 + smat * 16);
        kb[nt][2 * pr][0] = r0; kb[nt][2 * pr][1] = r1;
        kb[nt][2 * pr + 1][0] = r2; kb[nt][2 * pr + 1][1] = r3;
      }
    }
    uint32_t vb[4][4];
#pragma unroll
    for (int dp = 0; dp < 4; dp++)
      ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3],
             vf + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);

#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      if constexpr (QK8) {
#pragma unroll
        for (int ks = 0; ks < 2; ks++)
#pragma unroll
          for (int nt = 0; nt < 2; nt++)
            mma_e4m3(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      } else {
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[mt][ks][0], qa[mt][ks][1], qa[mt][ks][2], qa[mt][ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      }

      // masking (only on boundary stages), online softmax (two rows per thread: g and g+8)
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = mt * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;  // running max lives in the scaled log2 domain
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe);
        newm[h] = safe == 0.f && newm[h] == -INFINITY ? -INFINITY : newm[h];
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);  // row g
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);  // row g+8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
      // PV: A = P (a0: row g tokens 0-7, a1: row g+8 tokens 0-7, a2: row g tokens 8-15, a3: row g+8 tokens 8-15)
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[mt][2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[mt][2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();

  // ---- merge the four warps' states ----
  float* sm_m = reinterpret_cast<float*>(smem);                 // [4][BM]
  float* sm_l = sm_m + 4 * BM;                                  // [4][BM]
  float* sm_o = sm_l + 4 * BM;                                  // [4][BM][HD]
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = mt * 16 + g + 8 * h;
      if (t == 0) {
        sm_m[warp * BM + row] = m_run[mt][h];
        sm_l[warp * BM + row] = l_run[mt][h];
      }
#pragma unroll
      for (int d = 0; d < 8; d++) {
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t] = o[mt][d][2 * h];
        sm_o[(warp * BM + row) * HD + d * 8 + 2 * t + 1] = o[mt][d][2 * h + 1];
      }
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < BM * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mm = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mm = fmaxf(mm, sm_m[w * BM + row]);
    float ls = 0.f, acc = 0.f;
    if (mm != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * BM + row] == -INFINITY ? 0.f : exp2f(sm_m[w * BM + row] - mm);
        ls += sm_l[w * BM + row] * f;
        acc += sm_o[(w * BM + row) * HD + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mm + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// Verify blocks: 64 query rows (8 heads x 8 tokens) per KV head -> one warp per 16-row m-tile; the four warps share each
// 16-token K/V stage (converted once per CTA), so no cross-warp merge is needed.
template <int S>
__global__ void __launch_bounds__(128)
fd_partials_v(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int nstages = end > begin ? (end - begin + STAGE - 1) / STAGE : 0;

  constexpr int RAW = 2 * 16 * RP;
  uint8_t* ring = smem;
  uint8_t* kf = smem + S * RAW;            // [2][16 * KP]
  uint8_t* vf = kf + 2 * 16 * KP;          // [2][16 * KP]

  // Q fragments for this warp's m-tile (rows warp*16 + g and +8)
  uint32_t qa[4][4];
#pragma unroll
  for (int half = 0; half < 2; half++) {
    const int row = warp * 16 + g + 8 * half;
    const int token = row >> 3, qh = kh * 8 + (row & 7);
    const bool ok = token < L;
    const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      uint32_t lo = 0, hi = 0;
      if (ok) {
        uint32_t v0 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 2 * t);
        uint32_t v1 = *reinterpret_cast<const uint16_t*>(qp + 16 * ks + 8 + 2 * t);
        fp8x4_to_h2(v0 | (v1 << 16), lo, hi);
      }
      qa[ks][half ? 1 : 0] = lo;
      qa[ks][half ? 3 : 2] = hi;
    }
  }
  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int d = 0; d < 8; d++)
#pragma unroll
    for (int r = 0; r < 4; r++) o[d][r] = 0.f;

  const int which = tid >> 6, tk = (tid >> 2) & 15, c4 = tid & 3;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  auto row_of = [&](int tok) -> long {
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    return phys < 0 ? -1L : ((long)phys << pshift) + (tok & pmask);
  };
  auto load_row = [&](int s) -> long {
    const int tok = begin + s * STAGE + tk;
    return (s < nstages && tok < end) ? row_of(tok) : -1L;
  };
  auto issue = [&](int s, long r) {
    if (s < nstages) {
      const long rr = r < 0 ? 0 : r;
      const uint8_t* base = which ? VC : KC;
      cp16(ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16, base + (rr * HKV + kh) * HD + c4 * 16);
    }
    cp_commit();
  };
  {
    long pr[S];
#pragma unroll
    for (int s = 0; s < S; s++) pr[s] = load_row(s);
#pragma unroll
    for (int s = 0; s < S - 1; s++) issue(s, pr[s]);
    long nxt = pr[S - 1];
    const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
    for (int s = 0; s < nstages; s++) {
      cp_wait<S - 2>();
      issue(s + S - 1, nxt);
      nxt = load_row(s + S);
      // convert this thread's own chunk (K or V row segment) to f16
      {
        const uint8_t* src = ring + (s % S) * RAW + which * 16 * RP + tk * RP + c4 * 16;
        const uint4 u = *reinterpret_cast<const uint4*>(src);
        uint4 o0, o1;
        fp8x4_to_h2(u.x, o0.x, o0.y);
        fp8x4_to_h2(u.y, o0.z, o0.w);
        fp8x4_to_h2(u.z, o1.x, o1.y);
        fp8x4_to_h2(u.w, o1.z, o1.w);
        uint8_t* dst = (which ? vf : kf) + (s & 1) * 16 * KP + tk * KP + c4 * 32;
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 16) = o1;
      }
      __syncthreads();
      const uint8_t* kfs = kf + (s & 1) * 16 * KP;
      const uint8_t* vfs = vf + (s & 1) * 16 * KP;
      const int tb = begin + s * STAGE;
      const bool full_vis = (tb + STAGE <= end) && (L == 1 || tb + STAGE - 1 <= prefix) &&
                            (!window || L == 1 || tb > prefix + L - 1 - window);
      uint32_t kb[2][4][2];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int pr2 = 0; pr2 < 2; pr2++) {
          uint32_t r0, r1, r2, r3;
          ldsm4(r0, r1, r2, r3, kfs + (nt * 8 + srow) * KP + pr2 * 64 + smat * 16);
          kb[nt][2 * pr2][0] = r0; kb[nt][2 * pr2][1] = r1;
          kb[nt][2 * pr2 + 1][0] = r2; kb[nt][2 * pr2 + 1][1] = r3;
        }
      uint32_t vb[4][4];
#pragma unroll
      for (int dp = 0; dp < 4; dp++)
        ldsm4t(vb[dp][0], vb[dp][1], vb[dp][2], vb[dp][3], vfs + ((smat & 1) * 8 + srow) * KP + dp * 32 + (smat >> 1) * 16);
      float sacc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
          mma_f16(sacc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], kb[nt][ks][0], kb[nt][ks][1]);
      float mx[2] = {-INFINITY, -INFINITY};
      float sv[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) sv[nt][r] = sacc[nt][r];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = warp * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sv[nt][r] = -INFINITY;
          }
      }
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sv[nt][r]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
      }
      float corr[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        newm[h] = fmaxf(m_run[h], mx[h]);
        const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[h] - safe);
      }
      uint32_t pa[4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = r >> 1;
          const float safe = newm[h] == -INFINITY ? 0.f : newm[h];
          p[r] = exp2f(fmaf(sv[nt][r], sc2, -safe));
          rs[h] += p[r];
        }
        pa[nt * 2 + 0] = pack_h2(p[0], p[1]);
        pa[nt * 2 + 1] = pack_h2(p[2], p[3]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[h] = l_run[h] * corr[h] + rs[h];
        m_run[h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[d][0] *= corr[0]; o[d][1] *= corr[0];
          o[d][2] *= corr[1]; o[d][3] *= corr[1];
        }
      }
#pragma unroll
      for (int dp = 0; dp < 4; dp++) {
        mma_f16(o[2 * dp], pa[0], pa[1], pa[2], pa[3], vb[dp][0], vb[dp][1]);
        mma_f16(o[2 * dp + 1], pa[0], pa[1], pa[2], pa[3], vb[dp][2], vb[dp][3]);
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = l_run[h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int d = 0; d < 8; d++) {
      PART[(base_row + row) * HD + d * 8 + 2 * t] = o[d][2 * h] * inv * value_scale;
      PART[(base_row + row) * HD + d * 8 + 2 * t + 1] = o[d][2 * h + 1] * inv * value_scale;
    }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch_v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)S * 2 * 16 * RP + 4 * 16 * KP;
  cudaFuncSetAttribute(fd_partials_v<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials_v<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// fd2: decode (one query token, 8 heads per KV head) with 32-token stages.  K and V stay raw FP8 in shared memory:
// K fragments come straight from ldmatrix (e4m3 mma), V fragments from ldmatrix.trans + byte permutes + cvt.
// ---------------------------------------------------------------------------------------------------------------
#define ST2 32
#define RMAX 4096

__device__ __forceinline__ void mma_qk8(float* c, uint32_t qa0, uint32_t qa2, uint32_t b0, uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(qa0), "r"(0u), "r"(qa2), "r"(0u), "r"(b0), "r"(b1));
#else
  // exact emulation with f16 mma: each fp8 register holds 4 consecutive k, used as two k16 slots
  uint32_t ql, qh, kl, kh2;
  fp8x4_to_h2(qa0, ql, qh);
  fp8x4_to_h2(b0, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
  fp8x4_to_h2(qa2, ql, qh);
  fp8x4_to_h2(b1, kl, kh2);
  mma_f16(c, ql, 0u, qh, 0u, kl, kh2);
#endif
}
// two e4m3 bytes (low 16 bits of x) -> f16x2
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  uint32_t r;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}" : "=r"(r) : "r"(x));
  return r;
#else
  uint32_t lo, hi;
  fp8x4_to_h2(x, lo, hi);
  return lo;
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd2_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP;  // K and V raw rows of one stage
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  // physical cache row of every token of this CTA's range
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  pdl_wait();  // Q and the new cache rows come from the previous kernel
  // Q fragments: row g = query head kh*8+g, e4m3 bytes d = 32 ks + 4 t.. and +16
  uint32_t q0[2], q2[2];
  {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();  // rows[] ready

  float o[8][2];  // [d-tile (E0,O0,E1,O1,...)][row g, columns 2t, 2t+1]
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;  // offset within the CTA range
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    const int tb = begin + (warp + 4 * s) * ST2;
    // ---- S = Q K^T over 4 n-tiles of 8 tokens ----
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    // ---- softmax over the 32 columns of row g ----
    const bool full_vis = tb + ST2 <= end;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
    // ---- O += P V: two 16-token k-tiles ----
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        // pr selects d bytes [32 pr, 32 pr + 32): matrices (tok 0-7 | 8-15) x (m = 2 pr, 2 pr + 1)
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        // r0: (tok 0-7, m=2pr)  r1: (tok 8-15, m=2pr)  r2: (tok 0-7, m=2pr+1)  r3: (tok 8-15, m=2pr+1)
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();
  pdl_trigger();

  // ---- merge the four warps (rows g only: 8 query heads) ----
  float* sm_m = reinterpret_cast<float*>(smem);   // [4][8]
  float* sm_l = sm_m + 32;                        // [4][8]
  float* sm_o = sm_l + 32;                        // [4][8][64]
  if (t == 0) {
    sm_m[warp * 8 + g] = m_run;
    sm_l[warp * 8 + g] = l_run;
  }
#pragma unroll
  for (int mm = 0; mm < 4; mm++) {
    // tile 2mm = E (d = 16 mm + 2 n), tile 2mm+1 = O (d = 16 mm + 2 n + 1), n = 2t + c
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  }
  __syncthreads();
  for (int idx = tid; idx < 8 * HD; idx += 128) {
    const int row = idx / HD, d = idx % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[w * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float f = sm_m[w * 8 + row] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + row] - mmx);
        ls += sm_l[w * 8 + row] * f;
        acc += sm_o[(w * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}


// ---- fd3: one CTA serves BOTH kv heads (HKV == 2): rows are fetched as whole 128 B lines ------------------------
#define RP3 144   // bytes per K/V smem row: 128 + 16 pad

template <int S>
__global__ void __launch_bounds__(128)
fd3_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * ST2 * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + ST2 - 1) / ST2;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  pdl_wait();
  uint32_t q0[2][2], q2[2][2];
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();

  float o[2][8][2];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int kh = 0; kh < 2; kh++)
#pragma unroll
    for (int d = 0; d < 8; d++) o[kh][d][0] = o[kh][d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * ST2;
      const int c = lane & 7;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int tk = (lane >> 3) + 4 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = rr * 128 + c * 16;
        cp16(slot + tk * RP3 + c * 16, KC + goff);
        cp16(slot + ST2 * RP3 + tk * RP3 + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP3;
    const int tb = begin + (warp + 4 * s) * ST2;
    const bool full_vis = tb + ST2 <= end;
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {
      float sacc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
        uint32_t k0, k1, k2, k3;
        ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
        mma_qk8(sacc[nt], q0[kh][0], q2[kh][0], k0, k1);
        mma_qk8(sacc[nt], q0[kh][1], q2[kh][1], k2, k3);
      }
      float sv[4][2];
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          float v = sacc[nt][r];
          if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
          sv[nt][r] = v;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx *= sc2;
      const float newm = fmaxf(m_run[kh], mx);
      const float safe = newm == -INFINITY ? 0.f : newm;
      const float corr = exp2f(m_run[kh] - safe);
      float rs = 0.f;
      uint32_t pa[4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
        rs += p0 + p1;
        pa[nt] = pack_h2(p0, p1);
      }
      rs += __shfl_xor_sync(0xffffffffu, rs, 1);
      rs += __shfl_xor_sync(0xffffffffu, rs, 2);
      l_run[kh] = l_run[kh] * corr + rs;
      m_run[kh] = newm;
      if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) { o[kh][d][0] *= corr; o[kh][d][1] *= corr; }
      }
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
#pragma unroll
        for (int pr = 0; pr < 2; pr++) {
          uint32_t r0, r1, r2, r3;
          ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
          const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
          const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
          const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
          const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
          float acc[4];
          acc[0] = o[kh][4 * pr + 0][0]; acc[1] = o[kh][4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
          o[kh][4 * pr + 0][0] = acc[0]; o[kh][4 * pr + 0][1] = acc[1];
          acc[0] = o[kh][4 * pr + 1][0]; acc[1] = o[kh][4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
          o[kh][4 * pr + 1][0] = acc[0]; o[kh][4 * pr + 1][1] = acc[1];
          acc[0] = o[kh][4 * pr + 2][0]; acc[1] = o[kh][4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
          o[kh][4 * pr + 2][0] = acc[0]; o[kh][4 * pr + 2][1] = acc[1];
          acc[0] = o[kh][4 * pr + 3][0]; acc[1] = o[kh][4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
          o[kh][4 * pr + 3][0] = acc[0]; o[kh][4 * pr + 3][1] = acc[1];
        }
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();
  pdl_trigger();

  // merge the four warps for each kv head
  float* sm_m = reinterpret_cast<float*>(smem);   // [2][4][8]
  float* sm_l = sm_m + 64;                        // [2][4][8]
  float* sm_o = sm_l + 64;                        // [2][4][8][64]
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    if (t == 0) {
      sm_m[(kh * 4 + warp) * 8 + g] = m_run[kh];
      sm_l[(kh * 4 + warp) * 8 + g] = l_run[kh];
    }
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[kh][2 * mm][cc];
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[kh][2 * mm + 1][cc];
      }
  }
  __syncthreads();
  // Merge statistics once per query row, instead of recomputing four
  // exponentials for every output dimension. Each owner touches a disjoint row.
  if (tid < 16) {
    const int kh = tid / 8, row = tid % 8;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[(kh * 4 + w) * 8 + row]);
    float ls = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
      const int i = (kh * 4 + w) * 8 + row;
      const float mw = sm_m[i];
      const float f = mw == -INFINITY ? 0.f : exp2f(mw - mmx);
      ls += sm_l[i] * f;
      sm_m[i] = f;
    }
    sm_l[kh * 32 + row] = ls > 0.f ? 1.f / ls : 1.f;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
  __syncthreads();
  for (int idx = tid; idx < 2 * 8 * HD; idx += 128) {
    const int kh = idx / (8 * HD), rem = idx % (8 * HD);
    const int row = rem / HD, d = rem % HD;
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++)
      acc += sm_o[((kh * 4 + w) * 8 + row) * 64 + d] * sm_m[(kh * 4 + w) * 8 + row];
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    PART[(base_row + row) * HD + d] = acc * sm_l[kh * 32 + row] * value_scale;
  }
}

template <int S>
static void launch3(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP3;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(128 + 2 * 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd3_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  launch_pdl(fd3_partials<S>, grid, dim3(128), dyn, st, reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


template <int S, int TS>
__global__ void __launch_bounds__(128)
fd3t_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int HQ, int HKV,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 16;
  constexpr int SLOT = 2 * TS * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + 1);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int nstages = (n + TS - 1) / TS;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + 31) / 32) * 32;
  uint8_t* wbase = smem + rmax_rt * 4 + warp * (S * SLOT);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  pdl_wait();
  uint32_t q0[2][2], q2[2][2];
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    const uint8_t* qp = Q + ((long)b * HQ + kh * 8 + g) * HD;
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      q0[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
      q2[kh][ks] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
    }
  }
  __syncthreads();

  float o[2][8][2];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int kh = 0; kh < 2; kh++)
#pragma unroll
    for (int d = 0; d < 8; d++) o[kh][d][0] = o[kh][d][1] = 0.f;

  const int my_stages = (nstages - warp + 3) / 4 > 0 ? (nstages - warp + 3) / 4 : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + 4 * s) * TS;
      const int c = lane & 7;
#pragma unroll
      for (int i = 0; i < TS / 4; i++) {
        const int tk = (lane >> 3) + 4 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = rr * 128 + c * 16;
        cp16(slot + tk * RP3 + c * 16, KC + goff);
        cp16(slot + TS * RP3 + tk * RP3 + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    const uint8_t* slot = wbase + (s % S) * SLOT;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + TS * RP3;
    const int tb = begin + (warp + 4 * s) * TS;
    const bool full_vis = tb + TS <= end;
#pragma unroll
    for (int kh = 0; kh < 2; kh++) {
      float sacc[TS / 8][4];
#pragma unroll
      for (int nt = 0; nt < TS / 8; nt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
        uint32_t k0, k1, k2, k3;
        ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
        mma_qk8(sacc[nt], q0[kh][0], q2[kh][0], k0, k1);
        mma_qk8(sacc[nt], q0[kh][1], q2[kh][1], k2, k3);
      }
      float sv[TS / 8][2];
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < TS / 8; nt++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          float v = sacc[nt][r];
          if (!full_vis && tb + nt * 8 + 2 * t + r >= end) v = -INFINITY;
          sv[nt][r] = v;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx *= sc2;
      const float newm = fmaxf(m_run[kh], mx);
      const float safe = newm == -INFINITY ? 0.f : newm;
      const float corr = exp2f(m_run[kh] - safe);
      float rs = 0.f;
      uint32_t pa[TS / 8];
#pragma unroll
      for (int nt = 0; nt < TS / 8; nt++) {
        const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
        rs += p0 + p1;
        pa[nt] = pack_h2(p0, p1);
      }
      rs += __shfl_xor_sync(0xffffffffu, rs, 1);
      rs += __shfl_xor_sync(0xffffffffu, rs, 2);
      l_run[kh] = l_run[kh] * corr + rs;
      m_run[kh] = newm;
      if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) { o[kh][d][0] *= corr; o[kh][d][1] *= corr; }
      }
#pragma unroll
      for (int kk = 0; kk < TS / 16; kk++) {
#pragma unroll
        for (int pr = 0; pr < 2; pr++) {
          uint32_t r0, r1, r2, r3;
          ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
          const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
          const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
          const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
          const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
          float acc[4];
          acc[0] = o[kh][4 * pr + 0][0]; acc[1] = o[kh][4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
          o[kh][4 * pr + 0][0] = acc[0]; o[kh][4 * pr + 0][1] = acc[1];
          acc[0] = o[kh][4 * pr + 1][0]; acc[1] = o[kh][4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
          o[kh][4 * pr + 1][0] = acc[0]; o[kh][4 * pr + 1][1] = acc[1];
          acc[0] = o[kh][4 * pr + 2][0]; acc[1] = o[kh][4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
          o[kh][4 * pr + 2][0] = acc[0]; o[kh][4 * pr + 2][1] = acc[1];
          acc[0] = o[kh][4 * pr + 3][0]; acc[1] = o[kh][4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
          mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
          o[kh][4 * pr + 3][0] = acc[0]; o[kh][4 * pr + 3][1] = acc[1];
        }
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  __syncthreads();
  pdl_trigger();

  // merge the four warps for each kv head
  float* sm_m = reinterpret_cast<float*>(smem);   // [2][4][8]
  float* sm_l = sm_m + 64;                        // [2][4][8]
  float* sm_o = sm_l + 64;                        // [2][4][8][64]
#pragma unroll
  for (int kh = 0; kh < 2; kh++) {
    if (t == 0) {
      sm_m[(kh * 4 + warp) * 8 + g] = m_run[kh];
      sm_l[(kh * 4 + warp) * 8 + g] = l_run[kh];
    }
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[kh][2 * mm][cc];
        sm_o[((kh * 4 + warp) * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[kh][2 * mm + 1][cc];
      }
  }
  __syncthreads();
  for (int idx = tid; idx < 2 * 8 * HD; idx += 128) {
    const int kh = idx / (8 * HD), rem = idx % (8 * HD);
    const int row = rem / HD, d = rem % HD;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; w++) mmx = fmaxf(mmx, sm_m[(kh * 4 + w) * 8 + row]);
    float ls = 0.f, acc = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float mw = sm_m[(kh * 4 + w) * 8 + row];
        const float f = mw == -INFINITY ? 0.f : exp2f(mw - mmx);
        ls += sm_l[(kh * 4 + w) * 8 + row] * f;
        acc += sm_o[((kh * 4 + w) * 8 + row) * 64 + d] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    PART[(base_row + row) * HD + d] = acc * inv * value_scale;
    if (d == 0) LSE[base_row + row] = ls > 0.f ? (mmx + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S, int TS>
static void launch3t(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * TS * RP3;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(128 + 2 * 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd3t_partials<S, TS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  launch_pdl(fd3t_partials<S, TS>, grid, dim3(128), dyn, st, reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---- fd3v: verify blocks (8 tokens x 8 heads = 64 query rows per KV head), both KV heads per CTA, 128 B row fetches ----
// warp = (kv head, 32-row half); the four warps share each 64-token K/V tile; no cross-warp merge is needed.
#define T3V 64
static int g_fd2_dbg = 0;
extern "C" void fd2_set_dbg(int v) { g_fd2_dbg = v; }
__device__ __forceinline__ void mma_q16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
#if defined(FD_NATIVE) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  uint32_t gl, gh, hl, hh, bl, bh;
  fp8x4_to_h2(a0, gl, gh); fp8x4_to_h2(a1, hl, hh); fp8x4_to_h2(b0, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
  fp8x4_to_h2(a2, gl, gh); fp8x4_to_h2(a3, hl, hh); fp8x4_to_h2(b1, bl, bh);
  mma_f16(c, gl, hl, gh, hh, bl, bh);
#endif
}

template <int S>
__global__ void __launch_bounds__(128)
fd3v_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  constexpr int SLOTB = 2 * T3V * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = warp >> 1, mhalf = warp & 1;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int ntiles = (n + T3V - 1) / T3V;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + T3V - 1) / T3V) * T3V;
  uint8_t* ring = smem + rmax_rt * 4;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // Q fragments (e4m3 bytes): m-tile mg = 2 mhalf + mt, rows g and g + 8
  // f16 A fragments of the four k16 steps (QK runs on f16 mma: the e4m3 mma.sync is emulated on sm_90)
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    uint32_t xr[2][2][2];  // [half][ks][lo/hi segment]
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        xr[half][ks][0] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
        xr[half][ks][1] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int seg2 = 0; seg2 < 2; seg2++) {
        uint32_t gl, gh, hl, hh;
        fp8x4_to_h2(xr[0][ks][seg2], gl, gh);
        fp8x4_to_h2(xr[1][ks][seg2], hl, hh);
        qa[mt][2 * ks + seg2][0] = gl; qa[mt][2 * ks + seg2][1] = hl;
        qa[mt][2 * ks + seg2][2] = gh; qa[mt][2 * ks + seg2][3] = hh;
      }
  }
  __syncthreads();  // rows[] ready

  float o[2][8][4];
  float m_run[2][2], l_run[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    m_run[mt][0] = m_run[mt][1] = -INFINITY;
    l_run[mt][0] = l_run[mt][1] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++) o[mt][d][0] = o[mt][d][1] = o[mt][d][2] = o[mt][d][3] = 0.f;
  }

  auto issue = [&](int s) {
    if (s < ntiles) {
      uint8_t* slot = ring + (s % S) * SLOTB;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int ci = tid + 128 * i;
        const int which = ci >> 9, rem = ci & 511;
        const int row = rem >> 3, c = rem & 7;
        const int li = s * T3V + row;
        const long rr = li < n ? rows[li] : 0;
        if (!(dbg & 8)) cp16(slot + which * (T3V * RP3) + row * RP3 + c * 16, (which ? VC : KC) + rr * 128 + c * 16);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < ntiles; s++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(s + S - 1);
    const uint8_t* slot = ring + (s % S) * SLOTB;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + T3V * RP3;
    const int tb = begin + s * T3V;
    const bool full_vis = (tb + T3V <= end) && (tb + T3V - 1 <= prefix) && (!window || tb > prefix + L - 1 - window);
    uint32_t pa[2][8][2];
    float sall[2][8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      uint32_t kk4[4], bl[4], bh[4];
      if (!(dbg & 1)) {
      ldsm4(kk4[0], kk4[1], kk4[2], kk4[3], kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
#pragma unroll
      for (int j = 0; j < 4; j++) fp8x4_to_h2(kk4[j], bl[j], bh[j]);
      }
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int r = 0; r < 4; r++) sall[mt][nt][r] = (dbg & 1) ? (float)(nt + r) : 0.f;
        if (!(dbg & 1))
#pragma unroll
        for (int j = 0; j < 4; j++) mma_f16(sall[mt][nt], qa[mt][j][0], qa[mt][j][1], qa[mt][j][2], qa[mt][j][3], bl[j], bh[j]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      float (&sacc)[8][4] = sall[mt];
      if (!full_vis) {
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int row = (mhalf * 2 + mt) * 16 + g + 8 * (r >> 1);
            const int token = row >> 3;
            const int col = tb + nt * 8 + 2 * t + (r & 1);
            const int pos = prefix + token;
            const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
            if (!vis) sacc[nt][r] = -INFINITY;
          }
      }
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[nt][r]);
      float corr[2], safe[2], newm[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        mx[h] *= sc2;
        newm[h] = fmaxf(m_run[mt][h], mx[h]);
        safe[h] = newm[h] == -INFINITY ? 0.f : newm[h];
        corr[h] = exp2f(m_run[mt][h] - safe[h]);
      }
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        float p[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          p[r] = (dbg & 2) ? fmaf(sacc[nt][r], sc2, -safe[r >> 1]) : exp2f(fmaf(sacc[nt][r], sc2, -safe[r >> 1]));
          rs[r >> 1] += p[r];
        }
        pa[mt][nt][0] = pack_h2(p[0], p[1]);  // row g
        pa[mt][nt][1] = pack_h2(p[2], p[3]);  // row g + 8
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
        rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
        l_run[mt][h] = l_run[mt][h] * corr[h] + rs[h];
        m_run[mt][h] = newm[h];
      }
      if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
        for (int d = 0; d < 8; d++) {
          o[mt][d][0] *= corr[0]; o[mt][d][1] *= corr[0];
          o[mt][d][2] *= corr[1]; o[mt][d][3] *= corr[1];
        }
      }
    }
    // ---- O += P V: four 16-token k-steps; the V fragments are shared by both m-tiles ----
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      if (dbg & 4) break;
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          const uint32_t a0 = pa[mt][2 * kk][0], a1 = pa[mt][2 * kk][1], a2 = pa[mt][2 * kk + 1][0], a3 = pa[mt][2 * kk + 1][1];
          mma_f16(o[mt][4 * pr + 0], a0, a1, a2, a3, bE0, bE1);
          mma_f16(o[mt][4 * pr + 1], a0, a1, a2, a3, bO0, bO1);
          mma_f16(o[mt][4 * pr + 2], a0, a1, a2, a3, cE0, cE1);
          mma_f16(o[mt][4 * pr + 3], a0, a1, a2, a3, cO0, cO1);
        }
      }
    }
  }
  cp_wait<0>();
  // rows owned by this warp: no cross-warp merge.  Column mapping of the E/O d-tile pairs: d = 16 mm + 2 (2 t + c) + {0: E, 1: O}.
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = (mhalf * 2 + mt) * 16 + g + 8 * h;
      const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
      const float ls = l_run[mt][h];
      const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
      for (int mm = 0; mm < 4; mm++)
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc)] = o[mt][2 * mm][2 * h + cc] * inv * value_scale;
          PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc) + 1] = o[mt][2 * mm + 1][2 * h + cc] * inv * value_scale;
        }
      if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[mt][h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
    }
}


// ---- fd3w: like fd3v but 8 warps per CTA, one 16-row m-tile per warp (fewer registers -> more warps in flight) ----
template <int S>
__global__ void __launch_bounds__(256, 2)
fd3w_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
              const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
              const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ, int HKV,
              int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int BM = 64;
  constexpr int SLOTB = 2 * T3V * RP3;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = warp >> 2, mg = warp & 3;
  const int split = blockIdx.x, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const int n = end > begin ? end - begin : 0;
  const int ntiles = (n + T3V - 1) / T3V;

  int* rows = reinterpret_cast<int*>(smem);
  const int rmax_rt = ((chunk + T3V - 1) / T3V) * T3V;
  uint8_t* ring = smem + rmax_rt * 4;
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 256) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  uint32_t qa[4][4];
  {
    uint32_t xr[2][2][2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int row = mg * 16 + g + 8 * half;
      const int token = row >> 3, qh = kh * 8 + (row & 7);
      const uint8_t* qp = Q + ((long)(b * L + token) * HQ + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        xr[half][ks][0] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 4 * t);
        xr[half][ks][1] = *reinterpret_cast<const uint32_t*>(qp + 32 * ks + 16 + 4 * t);
      }
    }
#pragma unroll
    for (int ks = 0; ks < 2; ks++)
#pragma unroll
      for (int seg2 = 0; seg2 < 2; seg2++) {
        uint32_t gl, gh, hl, hh;
        fp8x4_to_h2(xr[0][ks][seg2], gl, gh);
        fp8x4_to_h2(xr[1][ks][seg2], hl, hh);
        qa[2 * ks + seg2][0] = gl; qa[2 * ks + seg2][1] = hl;
        qa[2 * ks + seg2][2] = gh; qa[2 * ks + seg2][3] = hh;
      }
  }
  __syncthreads();

  float o[8][4];
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;

  auto issue = [&](int s) {
    if (s < ntiles) {
      uint8_t* slot = ring + (s % S) * SLOTB;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ci = tid + 256 * i;
        const int which = ci >> 9, rem = ci & 511;
        const int row = rem >> 3, c = rem & 7;
        const int li = s * T3V + row;
        const long rr = li < n ? rows[li] : 0;
        if (!(dbg & 8)) cp16(slot + which * (T3V * RP3) + row * RP3 + c * 16, (which ? VC : KC) + rr * 128 + c * 16);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);

  const int srow = lane & 7, smat = lane >> 3;
#pragma unroll 1
  for (int s = 0; s < ntiles; s++) {
    cp_wait<S - 2>();
    __syncthreads();
    issue(s + S - 1);
    const uint8_t* slot = ring + (s % S) * SLOTB;
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + T3V * RP3;
    const int tb = begin + s * T3V;
    const bool full_vis = (tb + T3V <= end) && (tb + T3V - 1 <= prefix) && (!window || tb > prefix + L - 1 - window);
    float sacc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      if (!(dbg & 1)) {
        uint32_t kk4[4], bl[4], bh[4];
        ldsm4(kk4[0], kk4[1], kk4[2], kk4[3], kraw + (nt * 8 + srow) * RP3 + kh * 64 + smat * 16);
#pragma unroll
        for (int j = 0; j < 4; j++) fp8x4_to_h2(kk4[j], bl[j], bh[j]);
#pragma unroll
        for (int j = 0; j < 4; j++) mma_f16(sacc[nt], qa[j][0], qa[j][1], qa[j][2], qa[j][3], bl[j], bh[j]);
      } else {
        sacc[nt][0] = (float)nt;
      }
    }
    if (!full_vis) {
#pragma unroll
      for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int row = mg * 16 + g + 8 * (r >> 1);
          const int token = row >> 3;
          const int col = tb + nt * 8 + 2 * t + (r & 1);
          const int pos = prefix + token;
          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
          if (!vis) sacc[nt][r] = -INFINITY;
        }
    }
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
      for (int r = 0; r < 4; r++) mx[r >> 1] = fmaxf(mx[r >> 1], sacc[nt][r]);
    float corr[2], safe[2], newm[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      mx[h] *= sc2;
      newm[h] = fmaxf(m_run[h], mx[h]);
      safe[h] = newm[h] == -INFINITY ? 0.f : newm[h];
      corr[h] = exp2f(m_run[h] - safe[h]);
    }
    uint32_t pa[8][2];
    float rs[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      float p[4];
#pragma unroll
      for (int r = 0; r < 4; r++) {
        p[r] = (dbg & 2) ? fmaf(sacc[nt][r], sc2, -safe[r >> 1]) : exp2f(fmaf(sacc[nt][r], sc2, -safe[r >> 1]));
        rs[r >> 1] += p[r];
      }
      pa[nt][0] = pack_h2(p[0], p[1]);
      pa[nt][1] = pack_h2(p[2], p[3]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
      l_run[h] = l_run[h] * corr[h] + rs[h];
      m_run[h] = newm[h];
    }
    if (__any_sync(0xffffffffu, corr[0] != 1.f || corr[1] != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) {
        o[d][0] *= corr[0]; o[d][1] *= corr[0];
        o[d][2] *= corr[1]; o[d][3] *= corr[1];
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      if (dbg & 4) break;
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP3 + kh * 64 + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        const uint32_t a0 = pa[2 * kk][0], a1 = pa[2 * kk][1], a2 = pa[2 * kk + 1][0], a3 = pa[2 * kk + 1][1];
        mma_f16(o[4 * pr + 0], a0, a1, a2, a3, bE0, bE1);
        mma_f16(o[4 * pr + 1], a0, a1, a2, a3, bO0, bO1);
        mma_f16(o[4 * pr + 2], a0, a1, a2, a3, cE0, cE1);
        mma_f16(o[4 * pr + 3], a0, a1, a2, a3, cO0, cO1);
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = mg * 16 + g + 8 * h;
    const long base_row = ((long)(b * HKV + kh) * splits + split) * BM;
    const float ls = l_run[h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int mm = 0; mm < 4; mm++)
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][2 * h + cc] * inv * value_scale;
        PART[(base_row + row) * HD + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][2 * h + cc] * inv * value_scale;
      }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
}

template <int S>
static void launch3w(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)(((chunk + T3V - 1) / T3V) * T3V) * 4 + (size_t)S * 2 * T3V * RP3;
  cudaFuncSetAttribute(fd3w_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3w_partials<S><<<grid, 256, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled, g_fd2_dbg);
}

template <int S>
static void launch3v(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                     const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                     int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  const size_t dyn = (size_t)(((chunk + T3V - 1) / T3V) * T3V) * 4 + (size_t)S * 2 * T3V * RP3;
  cudaFuncSetAttribute(fd3v_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 1, B);
  fd3v_partials<S><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                          reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ, HKV,
                                          table_n, page, window, splits, chunk, sm_scale, scaled, g_fd2_dbg);
}

template <int S>
static void launch2(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                    const float* VS, float* PART, float* LSE, int B, int HQ, int HKV, int table_n, int page, int window,
                    int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)(((chunk + 31) / 32) * 32) * 4 + (size_t)4 * S * SLOT;
  const size_t merge = (size_t)(64 + 4 * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd2_partials<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  launch_pdl(fd2_partials<S>, grid, dim3(128), dyn, st, reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                         reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, HQ, HKV,
                                         table_n, page, window, splits, chunk, sm_scale, scaled);
}


// ---------------------------------------------------------------------------------------------------------------
// slide1: the whole sliding-window decode attention in ONE kernel (one query token, 8 heads per KV head):
// q/k RMS norm + RoPE, fp8 conversion, cache write of the new k/v, streamed attention over the window,
// cross-warp merge, per-head RMS norm and sigmoid gate.  grid (HKV, B), 256 threads.
// ---------------------------------------------------------------------------------------------------------------
#include <cuda_fp8.h>
#define SW 8                 // warps per CTA
#define SRMAX 2112           // max streamed tokens (window 2048 + slack)

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

template <int S>
__global__ void __launch_bounds__(SW * 32)
slide1_kernel(const bf16* __restrict__ QKV, const long* __restrict__ POS, const long* __restrict__ LOC,
              uint8_t* __restrict__ KC, uint8_t* __restrict__ VC, const int* __restrict__ TABLE,
              const int* __restrict__ SEQLEN, const float* __restrict__ KS, const float* __restrict__ VS,
              bf16* __restrict__ OUT, int HQ, int HKV, int table_n, int page, int window, float theta, float sm_scale,
              int scaled, int qkv_stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int SLOT = 2 * ST2 * RP;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int kh = blockIdx.x, b = blockIdx.y;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int n = prefix - first;            // streamed tokens (the new token is handled separately)
  const int nst = (n + ST2 - 1) / ST2;
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const bf16* qkv = QKV + (long)b * qkv_stride;
  const float pos = (float)POS[b];

  // shared layout: rows[SRMAX] | qs[8][64] | newslot (K row 0 / V row 0, zero filled) | per-warp rings | merge area
  int* rows = reinterpret_cast<int*>(smem);
  uint8_t* qs = smem + SRMAX * 4;
  uint8_t* newslot = qs + 512;                       // SLOT bytes
  uint8_t* rings = newslot + SLOT;                   // SW * S * SLOT bytes
  uint8_t* wbase = rings + warp * (S * SLOT);

  // ---- zero the new-token slot, fill the row table ----
  for (int i = tid; i < SLOT / 16; i += SW * 32) reinterpret_cast<uint4*>(newslot)[i] = make_uint4(0, 0, 0, 0);
  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += SW * 32) {
    const int tok = first + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  // ---- q: 8 query heads of this KV head (threads 0..127), k_new (warp 4), v_new (warp 5) ----
  if (tid < 128) {
    const int j = tid >> 4, part = tid & 15;
    const bf16* src = qkv + (kh * 8 + j) * HD + 4 * part;
    float v[4];
    {
      const uint2 raw = *reinterpret_cast<const uint2*>(src);
      const bf162* p = reinterpret_cast<const bf162*>(&raw);
      const float2 a = __bfloat1622float2(p[0]), c = __bfloat1622float2(p[1]);
      v[0] = a.x; v[1] = a.y; v[2] = c.x; v[3] = c.y;
    }
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
    for (int d = 1; d <= 8; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float inv = rsqrtf(ss / 64.f + 1e-5f);
    float o4[4];
#pragma unroll
    for (int k = 0; k < 4; k++) o4[k] = v[k] * inv;
    if (theta != 0.f) {
#pragma unroll
      for (int pp = 0; pp < 2; pp++) {
        const int i = 2 * part + pp;
        const float freq = expf((-2.f * (float)i / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = o4[2 * pp], x1 = o4[2 * pp + 1];
        o4[2 * pp] = x0 * cs - x1 * sn;
        o4[2 * pp + 1] = x0 * sn + x1 * cs;
      }
    }
    uint32_t w4 = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) w4 |= (uint32_t)f2e4m3(o4[k]) << (8 * k);
    *reinterpret_cast<uint32_t*>(qs + j * 64 + 4 * part) = w4;
  } else if (warp == 4 || warp == 5) {
    const int is_v = warp == 5;
    const bf16* src = qkv + (HQ + (is_v ? HKV : 0) + kh) * HD + 2 * lane;
    const bf162 raw = *reinterpret_cast<const bf162*>(src);
    float2 v = __bfloat1622float2(raw);
    if (!is_v) {
      float ss = v.x * v.x + v.y * v.y;
#pragma unroll
      for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
      const float inv = rsqrtf(ss / 64.f + 1e-5f);
      v.x *= inv; v.y *= inv;
      if (theta != 0.f) {
        const float freq = expf((-2.f * (float)lane / 64.f) * logf(theta));
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        const float x0 = v.x, x1 = v.y;
        v.x = x0 * cs - x1 * sn;
        v.y = x0 * sn + x1 * cs;
      }
      v.x /= key_scale; v.y /= key_scale;
    } else {
      v.x /= value_scale; v.y /= value_scale;
    }
    const uint8_t b0 = f2e4m3(v.x), b1 = f2e4m3(v.y);
    const long loc = LOC[b];
    uint8_t* dst = (is_v ? VC : KC) + (loc * HKV + kh) * HD + 2 * lane;
    *reinterpret_cast<uint16_t*>(dst) = (uint16_t)b0 | ((uint16_t)b1 << 8);
    uint8_t* ns = newslot + (is_v ? ST2 * RP : 0) + 2 * lane;     // row 0 of the new-token slot
    *reinterpret_cast<uint16_t*>(ns) = (uint16_t)b0 | ((uint16_t)b1 << 8);
  }
  __syncthreads();

  uint32_t q0[2], q2[2];
#pragma unroll
  for (int ks = 0; ks < 2; ks++) {
    q0[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 4 * t);
    q2[ks] = *reinterpret_cast<const uint32_t*>(qs + g * 64 + 32 * ks + 16 + 4 * t);
  }
  float o[8][2];
  float m_run = -INFINITY, l_run = 0.f;
#pragma unroll
  for (int d = 0; d < 8; d++) o[d][0] = o[d][1] = 0.f;

  const int my_stages = (nst - warp + SW - 1) / SW > 0 ? (nst - warp + SW - 1) / SW : 0;
  auto issue = [&](int s) {
    if (s < my_stages) {
      uint8_t* slot = wbase + (s % S) * SLOT;
      const int tb = (warp + SW * s) * ST2;
      const int c = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int tk = (lane >> 2) + 8 * i;
        const int li = tb + tk;
        const long rr = li < n ? rows[li] : 0;
        const long goff = (rr * HKV + kh) * HD + c * 16;
        cp16(slot + tk * RP + c * 16, KC + goff);
        cp16(slot + ST2 * RP + tk * RP + c * 16, VC + goff);
      }
    }
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < S - 1; s++) issue(s);
  const int srow = lane & 7, smat = lane >> 3;

  auto process = [&](const uint8_t* slot, int tb, int end_tok) {
    const uint8_t* kraw = slot;
    const uint8_t* vraw = slot + ST2 * RP;
    float sacc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
      for (int r = 0; r < 4; r++) sacc[nt][r] = 0.f;
      uint32_t k0, k1, k2, k3;
      ldsm4(k0, k1, k2, k3, kraw + (nt * 8 + srow) * RP + smat * 16);
      mma_qk8(sacc[nt], q0[0], q2[0], k0, k1);
      mma_qk8(sacc[nt], q0[1], q2[1], k2, k3);
    }
    const bool full_vis = tb + ST2 <= end_tok;
    float sv[4][2];
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
      for (int r = 0; r < 2; r++) {
        float v = sacc[nt][r];
        if (!full_vis && tb + nt * 8 + 2 * t + r >= end_tok) v = -INFINITY;
        sv[nt][r] = v;
        mx = fmaxf(mx, v);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx *= sc2;
    const float newm = fmaxf(m_run, mx);
    const float safe = newm == -INFINITY ? 0.f : newm;
    const float corr = exp2f(m_run - safe);
    float rs = 0.f;
    uint32_t pa[4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const float p0 = exp2f(fmaf(sv[nt][0], sc2, -safe)), p1 = exp2f(fmaf(sv[nt][1], sc2, -safe));
      rs += p0 + p1;
      pa[nt] = pack_h2(p0, p1);
    }
    rs += __shfl_xor_sync(0xffffffffu, rs, 1);
    rs += __shfl_xor_sync(0xffffffffu, rs, 2);
    l_run = l_run * corr + rs;
    m_run = newm;
    if (__any_sync(0xffffffffu, corr != 1.f)) {
#pragma unroll
      for (int d = 0; d < 8; d++) { o[d][0] *= corr; o[d][1] *= corr; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
#pragma unroll
      for (int pr = 0; pr < 2; pr++) {
        uint32_t r0, r1, r2, r3;
        ldsm4t(r0, r1, r2, r3, vraw + (kk * 16 + (smat & 1) * 8 + srow) * RP + pr * 32 + (smat >> 1) * 16);
        const uint32_t bE0 = cvt2(prmt(r0, 0u, 0x4420u)), bE1 = cvt2(prmt(r1, 0u, 0x4420u));
        const uint32_t bO0 = cvt2(prmt(r0, 0u, 0x4431u)), bO1 = cvt2(prmt(r1, 0u, 0x4431u));
        const uint32_t cE0 = cvt2(prmt(r2, 0u, 0x4420u)), cE1 = cvt2(prmt(r3, 0u, 0x4420u));
        const uint32_t cO0 = cvt2(prmt(r2, 0u, 0x4431u)), cO1 = cvt2(prmt(r3, 0u, 0x4431u));
        float acc[4];
        acc[0] = o[4 * pr + 0][0]; acc[1] = o[4 * pr + 0][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bE0, bE1);
        o[4 * pr + 0][0] = acc[0]; o[4 * pr + 0][1] = acc[1];
        acc[0] = o[4 * pr + 1][0]; acc[1] = o[4 * pr + 1][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, bO0, bO1);
        o[4 * pr + 1][0] = acc[0]; o[4 * pr + 1][1] = acc[1];
        acc[0] = o[4 * pr + 2][0]; acc[1] = o[4 * pr + 2][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cE0, cE1);
        o[4 * pr + 2][0] = acc[0]; o[4 * pr + 2][1] = acc[1];
        acc[0] = o[4 * pr + 3][0]; acc[1] = o[4 * pr + 3][1]; acc[2] = 0.f; acc[3] = 0.f;
        mma_f16(acc, pa[2 * kk], 0u, pa[2 * kk + 1], 0u, cO0, cO1);
        o[4 * pr + 3][0] = acc[0]; o[4 * pr + 3][1] = acc[1];
      }
    }
  };

#pragma unroll 1
  for (int s = 0; s < my_stages; s++) {
    cp_wait<S - 2>();
    __syncwarp();
    issue(s + S - 1);
    process(wbase + (s % S) * SLOT, first + (warp + SW * s) * ST2, prefix);
    __syncwarp();
  }
  cp_wait<0>();
  if (warp == 0) process(newslot, prefix, prefix + 1);   // the new token (position prefix), fp8-rounded as stored
  __syncthreads();

  // ---- merge the warps ----
  float* sm_m = reinterpret_cast<float*>(smem);   // reuses the row table: [SW][8], [SW][8], [SW][8][64]
  float* sm_l = sm_m + SW * 8;
  float* sm_o = sm_l + SW * 8;
  if (t == 0) { sm_m[warp * 8 + g] = m_run; sm_l[warp * 8 + g] = l_run; }
#pragma unroll
  for (int mm = 0; mm < 4; mm++)
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc)] = o[2 * mm][cc];
      sm_o[(warp * 8 + g) * 64 + 16 * mm + 2 * (2 * t + cc) + 1] = o[2 * mm + 1][cc];
    }
  __syncthreads();
  // warp h finalizes query head h of this KV head; lane owns dims 2 lane, 2 lane + 1
  {
    const int h = warp;
    float mmx = -INFINITY;
#pragma unroll
    for (int w = 0; w < SW; w++) mmx = fmaxf(mmx, sm_m[w * 8 + h]);
    float ls = 0.f, a0 = 0.f, a1 = 0.f;
    if (mmx != -INFINITY) {
#pragma unroll
      for (int w = 0; w < SW; w++) {
        const float f = sm_m[w * 8 + h] == -INFINITY ? 0.f : exp2f(sm_m[w * 8 + h] - mmx);
        ls += sm_l[w * 8 + h] * f;
        a0 += sm_o[(w * 8 + h) * 64 + 2 * lane] * f;
        a1 += sm_o[(w * 8 + h) * 64 + 2 * lane + 1] * f;
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
    // bf16 round, RMS norm over the 64 dims, sigmoid gate
    const float u0 = __bfloat162float(__float2bfloat16(a0 * inv * value_scale));
    const float u1 = __bfloat162float(__float2bfloat16(a1 * inv * value_scale));
    float ss = u0 * u0 + u1 * u1;
#pragma unroll
    for (int d = 1; d <= 16; d <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, d);
    const float rinv = rsqrtf(ss / 64.f + 1e-5f);
    const int qh = kh * 8 + h;
    const bf162 gate = *reinterpret_cast<const bf162*>(qkv + (HQ + 2 * HKV) * HD + qh * HD + 2 * lane);
    const float2 gf = __bfloat1622float2(gate);
    const float s0 = 1.f / (1.f + expf(-gf.x)), s1 = 1.f / (1.f + expf(-gf.y));
    *reinterpret_cast<bf162*>(OUT + ((long)b * HQ + qh) * HD + 2 * lane) = __floats2bfloat162_rn(u0 * rinv * s0, u1 * rinv * s1);
  }
}

template <int S>
static void launch_slide(const void* QKV, const void* POS, const void* LOC, void* KC, void* VC, const int* TABLE,
                         const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ, int HKV,
                         int table_n, int page, int window, float theta, float sm_scale, int scaled, int qkv_stride,
                         cudaStream_t st) {
  constexpr int SLOT = 2 * ST2 * RP;
  size_t dyn = (size_t)SRMAX * 4 + 512 + SLOT + (size_t)SW * S * SLOT;
  const size_t merge = (size_t)(2 * SW * 8 + SW * 8 * 64) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(slide1_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(HKV, B);
  slide1_kernel<S><<<grid, SW * 32, dyn, st>>>(reinterpret_cast<const bf16*>(QKV), reinterpret_cast<const long*>(POS),
                                              reinterpret_cast<const long*>(LOC), reinterpret_cast<uint8_t*>(KC),
                                              reinterpret_cast<uint8_t*>(VC), TABLE, SEQLEN, KS, VS,
                                              reinterpret_cast<bf16*>(OUT), HQ, HKV, table_n, page, window, theta,
                                              sm_scale, scaled, qkv_stride);
}

extern "C" int slide_run(int stages, const void* QKV, const void* POS, const void* LOC, void* KC, void* VC,
                         const int* TABLE, const int* SEQLEN, const float* KS, const float* VS, void* OUT, int B, int HQ,
                         int HKV, int table_n, int page, int window, float theta, float sm_scale, int scaled,
                         int qkv_stride, cudaStream_t st) {
#define SL(S_) if (stages == S_) { launch_slide<S_>(QKV, POS, LOC, KC, VC, TABLE, SEQLEN, KS, VS, OUT, B, HQ, HKV, table_n, page, window, theta, sm_scale, scaled, qkv_stride, st); return 0; }
  SL(2) SL(3) SL(4)
#undef SL
  return 1;
}

__global__ void pf_kernel(const char* p, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const size_t chunk = 32768;
  for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * chunk; off < bytes; off += (size_t)gridDim.x * blockDim.x * chunk) {
    const size_t n = bytes - off < chunk ? bytes - off : chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p + off), "r"((uint32_t)n) : "memory");
  }
#endif
}
extern "C" void pf_run(const void* p, size_t bytes, cudaStream_t st) { pf_kernel<<<4, 128, 0, st>>>(reinterpret_cast<const char*>(p), bytes); }

template <int MT, int S, bool QK8>
static void launch(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                   const float* VS, float* PART, float* LSE, int B, int L, int HQ, int HKV, int table_n, int page,
                   int window, int splits, int chunk, float sm_scale, int scaled, cudaStream_t st) {
  constexpr int BM = MT * 16;
  constexpr int PER_WARP = S * 2 * 16 * RP + 2 * 16 * KP;
  size_t dyn = 4 * PER_WARP;
  const size_t merge = (size_t)(8 * BM + 4 * BM * HD) * sizeof(float);
  if (merge > dyn) dyn = merge;
  cudaFuncSetAttribute(fd_partials<MT, S, QK8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, HKV, B);
  fd_partials<MT, S, QK8><<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                             reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                             HKV, table_n, page, window, splits, chunk, sm_scale, scaled);
}

extern "C" int fd_run(int mt, int stages, int qk8, const void* Q, const void* KC, const void* VC, const int* TABLE,
                      const int* SEQLEN, const float* KS, const float* VS, float* PART, float* LSE, int B, int L, int HQ,
                      int HKV, int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled,
                      cudaStream_t st) {
#define FD(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 0) { launch<MT_, S_, false>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
#define FD8(MT_, S_) if (mt == MT_ && stages == S_ && qk8 == 1) { launch<MT_, S_, true>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD(1, 3) FD(1, 4) FD(1, 6) FD(1, 8) FD(4, 3) FD(4, 4) FD(4, 6)
  FD8(1, 3) FD8(1, 4) FD8(1, 6) FD8(4, 3) FD8(4, 4) FD8(4, 6)
#undef FD8
#define FD2(S_) if (mt == 1 && stages == 200 + S_) { launch2<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD2(2) FD2(3) FD2(4)
#undef FD2
#define FD3W(S_) if (mt == 4 && HKV == 2 && L == 8 && stages == 500 + S_) { launch3w<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3W(2) FD3W(3) FD3W(4)
#undef FD3W
#define FD3V(S_) if (mt == 4 && HKV == 2 && L == 8 && stages == 400 + S_) { launch3v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3V(2) FD3V(3) FD3V(4)
#undef FD3V
#define FD3T(S_) if (mt == 1 && HKV == 2 && stages == 700 + S_) { launch3t<S_, 16>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3T(3) FD3T(4) FD3T(5) FD3T(6) FD3T(8)
#undef FD3T
#define FD3(S_) if (mt == 1 && HKV == 2 && stages == 300 + S_) { launch3<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FD3(2) FD3(3) FD3(4)
#undef FD3
#define FDV(S_) if (mt == 4 && stages == 100 + S_) { launch_v<S_>(Q, KC, VC, TABLE, SEQLEN, KS, VS, PART, LSE, B, L, HQ, HKV, table_n, page, window, splits, chunk, sm_scale, scaled, st); return 0; }
  FDV(3) FDV(4) FDV(6) FDV(8)
#undef FDV
#undef FD
  return 1;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int slide_run(int, const void*, const void*, const void*, void*, void*, const int*, const int*, const float*,
                         const float*, void*, int, int, int, int, int, int, float, float, int, int, cudaStream_t);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled) {
  int r = slide_run((int)stages, qkv.data_ptr(), pos.data_ptr(), loc.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                    seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), out.data_ptr(), (int)seqlens.numel(),
                    (int)HQ, (int)HKV, (int)table.size(1), (int)page, (int)window, (float)theta, (float)sm_scale, (int)scaled,
                    (int)qkv.stride(0), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported slide config");
}
extern "C" void fd2_set_dbg(int);
void fd2_set_dbg_t(int64_t v) { fd2_set_dbg((int)v); }
extern "C" void fd_set_pdl(int);
void fd_set_pdl_t(int64_t v) { fd_set_pdl((int)v); }
extern "C" void pf_run(const void*, size_t, cudaStream_t);
void pf_t(torch::Tensor w) { pf_run(w.data_ptr(), (size_t)w.numel() * w.element_size(), at::cuda::getCurrentCUDAStream()); }
extern "C" int fd_run(int, int, int, const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*,
                      float*, int, int, int, int, int, int, int, int, int, float, int, cudaStream_t);

void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled) {
  int r = fd_run((int)mt, (int)stages, (int)qk8, q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(),
                 seqlens.data_ptr<int>(), ks.data_ptr<float>(), vs.data_ptr<float>(), part.data_ptr<float>(),
                 lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ, (int)HKV, (int)table.size(1), (int)page,
                 (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "unsupported fd config");
}
'''
_FD_CPP = r'''
void pf_t(torch::Tensor w);
void fd2_set_dbg_t(int64_t v);
void fd_set_pdl_t(int64_t v);
void slide_run_t(int64_t stages, torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
                 torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t HQ,
                 int64_t HKV, int64_t page, int64_t window, double theta, double sm_scale, int64_t scaled);
void fd_run_t(int64_t mt, int64_t stages, int64_t qk8, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table,
              torch::Tensor seqlens, torch::Tensor ks, torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L,
              int64_t HQ, int64_t HKV, int64_t page, int64_t window, int64_t splits, int64_t chunk, double sm_scale,
              int64_t scaled);
'''

_FD_MOD = None
_FD_FLAGS = ["-DFD_HW_CVT", "-DFD_NATIVE"]
_FD_LAB = False
_FD_PDL = True
_PDL_MAXB = 1
# policy: which shapes use the CUDA split-KV kernel, with (stages, splits_fn)
_FD_POLICY = {('full', False): [(2, 24, 2, 64, 3), (25, 64, 2, 32, 3), (1, 1, 2, 128, 2), (2, 24, 2, 64, 2), (25, 64, 3, 32, 2)], ('sliding', False): [(1, 1, 3, 16, 2), (2, 24, 3, 4, 2), (25, 64, 3, 2, 2)]}


def _fd_module():
    global _FD_MOD
    if _FD_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _FD_MOD = load_inline("muse_fd_attn", _FD_CPP, cuda_sources=_FD_CUDA, functions=["fd_run_t", "slide_run_t", "pf_t", "fd2_set_dbg_t", "fd_set_pdl_t"],
                              extra_cuda_cflags=["-O3"] + _FD_FLAGS, verbose=False)
        _FD_MOD.fd_set_pdl_t(int(_FD_PDL and torch.cuda.get_device_capability()[0] >= 9))
    return _FD_MOD


def _fd_forward(data, q, qkv, stages, splits, block=16, qk8=0, variant=1):
    cfg = data['config']
    use_pdl = _FD_PDL and cfg['batch'] <= _PDL_MAXB
    _fd_module().fd_set_pdl_t(int(use_pdl and torch.cuda.get_device_capability()[0] >= 9))
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    scaled = cfg['kv_scale'] != 1.0
    window = cfg['window']
    span = window + length - 1 if window else data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), block) * block
    if variant in (2, 3, 4, 5, 6) and chunk > 4096:
        raise ValueError('chunk too large for fd2')
    bm = max(16, triton.next_power_of_2(length * (hq // hkv)))
    partial = torch.empty((batch, hkv, splits, bm, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, bm), device=qkv.device, dtype=torch.float32)
    _fd_module().fd_run_t(bm // 16, stages + (700 if variant == 6 else 500 if variant == 5 else 400 if variant == 4 else 100 if bm == 64 else 200 if variant == 2 else 300 if variant == 3 else 0), qk8, q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
                          data['k_scale'], data['v_scale'], partial, lse, length, hq, hkv, cfg['page_size'], window,
                          splits, chunk, cfg['sm_scale'] * cfg['qk_scale_factor'], int(scaled))
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, bm, num_warps=4, launch_pdl=use_pdl)
    return result


def _slide_forward(data, stages, qkv=None):
    cfg = data['config']
    batch = cfg['batch']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    qkv = (data['qkv'] if qkv is None else qkv).view(batch, -1)
    out = torch.empty((batch, 1, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _fd_module().slide_run_t(stages, qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'],
                             data['page_table'], data['seq_lens'], data['k_scale'], data['v_scale'], out, hq, hkv,
                             cfg['page_size'], cfg['window'], float(cfg['rope_theta']),
                             cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0))
    return out


def _fd_prepare(data):
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


def _fd_fits(cfg, table_n, splits, variant):
    if variant in (3, 4, 5, 6) and cfg['num_kv_heads'] != 2:
        return False
    if variant in (4, 5) and cfg['q_len'] != 8:
        return False
    if variant not in (2, 3, 4, 5, 6):
        return True
    span = cfg['window'] + cfg['q_len'] - 1 if cfg['window'] else table_n * cfg['page_size']
    return -(-(-(-span // splits)) // 16) * 16 <= 4096


def _fd_choice(cfg, context_bound):
    """(stages, splits) when the CUDA kernel should run this shape, else None."""
    key = ('sliding' if cfg['window'] else 'full', cfg['q_len'] > 1)
    for entry in _FD_POLICY.get(key, ()):
        lo_b, hi_b, stages, splits = entry[:4]
        variant = entry[4] if len(entry) > 4 else 1
        if lo_b <= cfg['batch'] <= hi_b and _fd_fits(cfg, context_bound // cfg['page_size'], splits, variant):
            return stages, splits, variant
    return None


def kernel_fn(data):
    cfg = data['config']
    ctx_bound = data['page_table'].shape[1] * cfg['page_size']
    choice = _fd_choice(cfg, ctx_bound)
    if _FD_LAB and not torch.cuda.is_current_stream_capturing() and not getattr(kernel_fn, '_labbed', False):
        kernel_fn._labbed = True
        _fd_lab()
    if (cfg['q_len'] == 1 and 8 <= cfg['batch'] <= 64 and cfg['num_q_heads'] == 16
            and cfg['num_kv_heads'] == 2 and ctx_bound <= (1 << 20) + 64
            and cfg['page_size'] & (cfg['page_size'] - 1) == 0):
        qkv, q = _fd_prepare(data)
        return _sp_forward(data, q, qkv)
    if (cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] <= 4
            and cfg['num_kv_heads'] == 2 and cfg['num_q_heads'] == 16
            and 32768 <= ctx_bound <= (1 << 20) + 64):
        qkv, q = _fd_prepare(data)
        splits = max(32, 128 // triton.next_power_of_2(cfg['batch']))
        return _wg_small_forward(data, q, qkv, splits)
    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] >= _WG_MINB and cfg['num_kv_heads'] == 2
            and cfg['num_q_heads'] == 16 and ctx_bound <= (1 << 20) + 64):
        qkv, q = _fd_prepare(data)
        return _wg_forward(data, q, qkv, 16 if cfg['batch'] <= 24 else 32)
    if choice is None or ctx_bound > (1 << 20) + 64:
        return _v7_kernel_fn(data)
    stages, splits, variant = choice
    qkv, q = _fd_prepare(data)
    return _fd_forward(data, q, qkv, stages, splits, variant=variant)


# Small full-context verification: independent-head 128-key tiles.
_WG_SMALL_CUDA = r'''
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define HD 64
#define WT 128              // keys per tile

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t cvt_fp8x2(uint32_t x) {   // low two e4m3 bytes -> f16x2
  uint32_t r;
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}" : "=r"(r) : "r"(x));
  return r;
}

__device__ __forceinline__ void wg_qk(float* d, uint64_t da, uint64_t db, int accum) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db), "r"(accum));
#endif
}

__device__ __forceinline__ void wg_pv(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void wg_pv72(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}
__device__ __forceinline__ void wg_pvt(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}


__device__ __forceinline__ void wg_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum);
__device__ __forceinline__ void wg_qk128(float* d, uint64_t da, uint64_t db, int accum) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(da), "l"(db), "r"(accum));
#endif
}
__device__ __forceinline__ uint64_t mkdesc(const void* p, uint32_t lbo, uint32_t sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit_wait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// core-matrix layout of a [rows][64 byte] fp8 tile: 8 rows x 16 B cores, K cores 128 B apart, row groups 512 B apart
__device__ __forceinline__ int core_off(int r, int chunk) { return (r >> 3) * 512 + chunk * 128 + (r & 7) * 16; }
// vT: [d][key] f16, cores of 8 d-rows x 8 keys (16 B per row): key cores 128 B apart, d groups 1024 B apart
__device__ __forceinline__ int vt_off(int d, int key) { return (d >> 3) * 1024 + (key >> 3) * 128 + (d & 7) * 16 + (key & 7) * 2; }

__device__ __forceinline__ int vmn_off(int key, int d) { return (key >> 3) * 1152 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2; }
#define WS 2   // raw stages
#define TILEB 8192   // bytes of a raw K or V tile
__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum) { wg_qk128(d, da, db, accum); }


__device__ __forceinline__ void wg_qk_first(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 0, 1, 1;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
               : "l"(da), "l"(db));
#endif
}
__device__ __forceinline__ void wg_qk_acc(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(da), "l"(db));
#endif
}

#define LAZY_TH 8.0f
__global__ void __launch_bounds__(128, 2)
wg5_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,
             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,
             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ,
             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, wgi = 0, tw = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;
  const int g = lane >> 2, t = lane & 3;
  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
  const int prefix = SEQLEN[b];
  const int first = window ? max(prefix - window + 1, 0) : 0;
  const int begin = first + split * chunk;
  const int end = min(begin + chunk, prefix + L);
  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;
  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;
  const long base_row = ((long)(b * 2 + kh) * splits + split) * 64;
  const int n = end > begin ? end - begin : 0;
  const int nst = (n + WT - 1) / WT;

  uint8_t* qs = smem;
  uint8_t* vt = smem + 4096;
  uint8_t* stages = smem + 4096 + 18432;
  int* rows = reinterpret_cast<int*>(stages + WS * 16384);

  const int pshift = __ffs(page) - 1, pmask = page - 1;
  for (int i = tid; i < n; i += 128) {
    const int tok = begin + i;
    const int pg = tok >> pshift;
    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;
    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
  }
  for (int c = tw; c < 64 * 4; c += 128) {
    const int r = c >> 2, ch = c & 3;
    const int token = r >> 3, qh = kh * 8 + (r & 7);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (token < L) v = *reinterpret_cast<const uint4*>(Q + ((long)(b * L + token) * HQ + qh) * HD + ch * 16);
    *reinterpret_cast<uint4*>(qs + core_off(r, ch)) = v;
  }
  {
    const uint32_t ad = (uint32_t)__cvta_generic_to_shared(vt + (tw >> 3) * 1152 + 1024 + (tw & 7) * 16);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ad), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u), "r"(0x3C003C00u));
  }
  proxy_fence();
  __syncthreads();

  // ---- loop-invariant addressing ----
  const uint32_t stages_s = (uint32_t)__cvta_generic_to_shared(stages);
  const int ci = tid & 7;                   // 16-byte chunk within the key row (K 0-7, V 8-15)
  const int cc = ci & 3;
  const bool isv = ci >= 4;
  const uint8_t* srcb = (isv ? VC : KC) + kh * 64 + cc * 16;
  uint32_t dst_off[8];
  int key4[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int key = (tid >> 3) + 16 * i;
    key4[i] = key;
    dst_off[i] = isv ? (8192 + key * 64 + (cc ^ (key & 3)) * 16) : core_off(key, cc);
  }
  auto issue = [&](int s, int stage) {
    if (s < nst && !(dbg & 8)) {
      const uint32_t base = stages_s + stage * 16384;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int li = s * WT + key4[i];
        const long rr = li < n ? rows[li] : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(base + dst_off[i]), "l"(srcb + rr * 128));
      }
    }
    cp_commit();
  };
  int wr_stage = 0;
#pragma unroll
  for (int s = 0; s < WS - 1; s++) { issue(s, wr_stage); wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1; }

  const uint64_t qd0 = mkdesc(qs, 128, 512), qd1 = mkdesc(qs + 256, 128, 512);
  const uint64_t kd0 = mkdesc(stages, 128, 512);
  const uint64_t vd0 = mkdesc(vt, 1152, 128);
  // per-thread V conversion offsets (loop invariant)
  // rotated chunk assignment: lanes take consecutive keys but different 16 B chunks, so both the smem loads and the stores are (nearly) conflict free
  const int vkey = tw;
  const int vcb = 0;
  uint32_t vsrc_off[4], vdst_off[8];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int cs = vcb + j;
    vsrc_off[j] = vkey * 64 + (cs ^ (vkey & 3)) * 16;
    vdst_off[2 * j] = (uint32_t)vmn_off(vkey, 16 * cs);
    vdst_off[2 * j + 1] = (uint32_t)vmn_off(vkey, 16 * cs + 8);
  }

  float o[36];
#pragma unroll
  for (int i = 0; i < 36; i++) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY};

  int rd_stage = 0;
#pragma unroll 1
  for (int s = 0; s < nst; s++) {
    cp_wait<WS - 2>();
    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }
    issue(s + WS - 1, wr_stage);
    wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1;
    const uint8_t* vst = stages + rd_stage * 16384 + 8192;
    const uint64_t kdesc = kd0 + (uint64_t)(rd_stage * 1024);
    rd_stage = rd_stage + 1 == WS ? 0 : rd_stage + 1;

    float sacc[64] = {};
    wg_fence();
    wg_qk128(sacc, qd0, kdesc, 0);
    wg_qk128(sacc, qd1, kdesc + 16, 1);
    wg_commit();
    // V raw [key][d] fp8 -> MN-major f16 core layout; overlaps with the QK wgmma
#pragma unroll
    for (int h2 = 0; h2 < ((dbg & 16) ? 0 : 4); h2++) {
      const uint4 u = *reinterpret_cast<const uint4*>(vst + vsrc_off[h2]);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q4 = 0; q4 < 2; q4++) {
        uint4 o4;
        o4.x = cvt_fp8x2(w[2 * q4]); o4.y = cvt_fp8x2(w[2 * q4] >> 16);
        o4.z = cvt_fp8x2(w[2 * q4 + 1]); o4.w = cvt_fp8x2(w[2 * q4 + 1] >> 16);
        *reinterpret_cast<uint4*>(vt + vdst_off[2 * h2 + q4]) = o4;
      }
    }
    proxy_fence();
    wg_wait0();
#pragma unroll
    for (int i = 0; i < 64; i++) asm volatile("" : "+f"(sacc[i])::"memory");
    asm volatile("bar.sync %0, 128;\n" ::"r"(1 + wgi) : "memory");

    const int tb = begin + s * WT;
    const bool full_vis = (tb + WT <= end) && (L == 1 || tb + WT - 1 <= prefix) && (!window || L == 1 || tb > prefix + L - 1 - window);
    if (!full_vis) {
#pragma unroll
      for (int j = 0; j < 16; j++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int row = warp * 16 + g + 8 * (r >> 1);
          const int token = row >> 3;
          const int col = tb + j * 8 + 2 * t + (r & 1);
          const int pos = prefix + token;
          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);
          if (!vis) sacc[4 * j + r] = -INFINITY;
        }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < ((dbg & 64) ? 0 : 16); j++) {
      mx0 = fmaxf(mx0, fmaxf(sacc[4 * j], sacc[4 * j + 1]));
      mx1 = fmaxf(mx1, fmaxf(sacc[4 * j + 2], sacc[4 * j + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    mx0 *= sc2; mx1 *= sc2;
    // lazy rescale: the reference max only moves when a tile exceeds it by more than LAZY_TH (log2 domain), so p <= 2^LAZY_TH
    if (__any_sync(0xffffffffu, mx0 > m_run[0] + LAZY_TH || mx1 > m_run[1] + LAZY_TH)) {
      const float n0 = fmaxf(m_run[0], mx0), n1 = fmaxf(m_run[1], mx1);
      const float s0 = n0 == -INFINITY ? 0.f : n0, s1 = n1 == -INFINITY ? 0.f : n1;
      const float c0 = ex2f(m_run[0] - s0), c1 = ex2f(m_run[1] - s1);
      m_run[0] = n0; m_run[1] = n1;
#pragma unroll
      for (int j = 0; j < 9; j++) {
        o[4 * j] *= c0; o[4 * j + 1] *= c0;
        o[4 * j + 2] *= c1; o[4 * j + 3] *= c1;
      }
    }
    const float sf0 = m_run[0] == -INFINITY ? 0.f : m_run[0], sf1 = m_run[1] == -INFINITY ? 0.f : m_run[1];
    uint32_t pa[16][2];
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const float p0 = (dbg & 32) ? sacc[4 * j] : ex2f(fmaf(sacc[4 * j], sc2, -sf0)), p1 = (dbg & 32) ? sacc[4 * j + 1] : ex2f(fmaf(sacc[4 * j + 1], sc2, -sf0));
      const float p2 = (dbg & 32) ? sacc[4 * j + 2] : ex2f(fmaf(sacc[4 * j + 2], sc2, -sf1)), p3 = (dbg & 32) ? sacc[4 * j + 3] : ex2f(fmaf(sacc[4 * j + 3], sc2, -sf1));
      pa[j][0] = pack_h2(p0, p1);
      pa[j][1] = pack_h2(p2, p3);
    }
    wg_fence();
    const uint64_t vd = vd0;
#pragma unroll
    for (int kk = 0; kk < 8; kk++)
      if (!(dbg & 2)) wg_pv72(o, pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1], vd + (uint64_t)(kk * 144));
    wg_commit_wait();
#pragma unroll
    for (int i = 0; i < 36; i++) asm volatile("" : "+f"(o[i])::"memory");
  }
  cp_wait<0>();
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int row = warp * 16 + g + 8 * h;
    const float ls = o[32 + 2 * h];
    const float inv = ls > 0.f ? 1.f / ls : 1.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      PART[(base_row + row) * HD + j * 8 + 2 * t] = o[4 * j + 2 * h] * inv * value_scale;
      PART[(base_row + row) * HD + j * 8 + 2 * t + 1] = o[4 * j + 2 * h + 1] * inv * value_scale;
    }
    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls)) * 0.6931471805599453f : -INFINITY;
  }
#endif
}

extern "C" int wg5_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,
                       const float* VS, float* PART, float* LSE, int B, int L, int HQ, int table_n, int page,
                       int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {
  const size_t rmax = ((chunk + 255) / 256) * 256;
  const size_t dyn = 1024 + 4096 + 18432 + (size_t)WS * 16384 + rmax * 4;
  cudaFuncSetAttribute(wg5_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
  dim3 grid(splits, 2, B);
  wg5_partials<<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),
                                       reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,
                                       table_n, page, window, splits, chunk, sm_scale, scaled, dbg);
  return 0;
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
extern "C" int wg5_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,
                       int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);
void wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,
               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,
               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {
  int r = wg5_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),
                  vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,
                  (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,
                  at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "wg5");
}
'''

_WG_SMALL_MOD = None

def _wg_small_module():
    global _WG_SMALL_MOD
    if _WG_SMALL_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _WG_SMALL_MOD = load_inline("muse_wg5_onehead_tile128", _WG_CPP, cuda_sources=_WG_SMALL_CUDA, functions=["wg5_run_t"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _WG_SMALL_MOD


def _wg_small_forward(data, q, qkv, splits):
    cfg = data['config']
    batch, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = batch * length
    span = data['page_table'].shape[1] * cfg['page_size']
    chunk = triton.cdiv(triton.cdiv(span, splits), 128) * 128
    partial = torch.empty((batch, hkv, splits, 64, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, 64), device=qkv.device, dtype=torch.float32)
    _wg_small_module().wg5_run_t(q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'], data['k_scale'],
                           data['v_scale'], partial, lse, length, hq, cfg['page_size'], 0, splits, chunk,
                           cfg['sm_scale'] * cfg['qk_scale_factor'], int(cfg['kv_scale'] != 1.0), 0)
    result = torch.empty((batch, length, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _merge_norm_gate[(tokens, hq)](partial, lse, qkv, result, hq, hkv, d, length, qkv.stride(0), splits, 64, num_warps=4)
    return result



_SP_CPP = 'void sparse_decode_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,int64_t);'
_SP_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n__device__ __forceinline__ int off(int key,int byte){return key*128+(((byte>>4)^(key&7))<<4)+(byte&15);}\n__device__ __forceinline__ uint32_t cv2(uint32_t x){uint32_t y;asm("{.reg .b16 a,b; mov.b32 {a,b},%1; cvt.rn.f16x2.e4m3x2 %0,a;}":"=r"(y):"r"(x));return y;}\n__device__ __forceinline__ void ld(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void qmma(float* c,uint32_t* a,uint32_t b0,uint32_t b1){asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));}\n__device__ __forceinline__ void pmma(float* c,uint32_t* a,uint32_t* b){asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}\n\ntemplate<int ST>\n__global__ void __launch_bounds__(128,2) trans_decode(const uint8_t* Q,const uint8_t* KC,const uint8_t* VC,\n const int* TABLE,const int* SEQ,const float* KS,const float* VS,float* PART,float* LSE,\n int table_n,int page,int window,int splits,int chunk,float sm_scale,int scaled){\n extern __shared__ __align__(16) uint8_t mem[];\n int tid=threadIdx.x,lane=tid&31,warp=tid>>5,h=warp&1,part=warp>>1,g=lane>>2,t=lane&3,b=blockIdx.y,split=blockIdx.x;\n int prefix=SEQ[b],first=window?max(0,prefix-window+1):0,begin=first+split*chunk;\n int end=min(begin+chunk,prefix+1),n=max(0,end-begin),tiles=(n+31)/32;\n int* rows=reinterpret_cast<int*>(mem+ST*8192);\n int shift=__ffs(page)-1,mask=page-1;\n for(int i=tid;i<n;i+=128){int tok=begin+i,pg=tok>>shift,phys=pg<table_n?TABLE[b*table_n+pg]:-1;rows[i]=phys<0?0:(phys<<shift)+(tok&mask);}\n uint32_t qb[2][2];\n#pragma unroll\n for(int k=0;k<2;k++){const uint8_t* q=Q+(long)b*1024+(h*8+g)*64+k*32+t*4;qb[k][0]=*reinterpret_cast<const uint32_t*>(q);qb[k][1]=*reinterpret_cast<const uint32_t*>(q+16);}\n float scale=sm_scale*(scaled?KS[0]:1.f)*1.4426950408889634f,vscale=scaled?VS[0]:1.f;\n __syncthreads();\n auto issue=[&](int tile){if(tile<tiles){uint8_t* dst=mem+(tile%ST)*8192;\n#pragma unroll\n for(int i=tid;i<256;i+=128){int key=i>>3,byte=(i&7)*16,li=tile*32+key;long rr=li<n?rows[li]:0;uint32_t ka=__cvta_generic_to_shared(dst+off(key,byte)),va=__cvta_generic_to_shared(dst+4096+off(key,byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(ka),"l"(KC+rr*128+byte));asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(va),"l"(VC+rr*128+byte));}}\n asm volatile("cp.async.commit_group;");};\n#pragma unroll\n for(int i=0;i<ST-1;i++)issue(i);\n float o[4][4]={},mm[2]={-INFINITY,-INFINITY},ll[2]={0.f,0.f};\n#pragma unroll 1\n for(int tile=0;tile<tiles;tile++){\n  asm volatile("cp.async.wait_group %0;"::"n"(ST-2));__syncthreads();issue(tile+ST-1);\n  const uint8_t* raw=mem+(tile%ST)*8192;float s[4]={};\n#pragma unroll\n  for(int k=0;k<2;k++){uint32_t a[4];ld(a,raw+off(part*16+(lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));qmma(s,a,qb[k][0],qb[k][1]);}\n#pragma unroll\n  for(int r=0;r<4;r++)s[r]=(tile*32+part*16+g+(r>=2?8:0)<n)?s[r]*scale:-INFINITY;\n  float mx[2]={fmaxf(s[0],s[2]),fmaxf(s[1],s[3])};\n#pragma unroll\n  for(int c=0;c<2;c++){\n#pragma unroll\n   for(int x=4;x<=16;x*=2)mx[c]=fmaxf(mx[c],__shfl_xor_sync(0xffffffff,mx[c],x));\n   float nm=fmaxf(mm[c],mx[c]),safe=nm==-INFINITY?0.f:nm,corr=exp2f(mm[c]-safe);ll[c]*=corr;mm[c]=nm;\n#pragma unroll\n   for(int d=0;d<4;d++){o[d][c]*=corr;o[d][c+2]*=corr;}\n  }\n  float p[4];\n#pragma unroll\n  for(int r=0;r<4;r++){float safe=mm[r&1]==-INFINITY?0.f:mm[r&1];p[r]=exp2f(s[r]-safe);}\n#pragma unroll\n  for(int c=0;c<2;c++){float sum=p[c]+p[c+2];\n#pragma unroll\n   for(int x=4;x<=16;x*=2)sum+=__shfl_xor_sync(0xffffffff,sum,x);ll[c]+=sum;}\n  __half2 hp0=__floats2half2_rn(p[0],p[1]),hp8=__floats2half2_rn(p[2],p[3]);\n  uint32_t p0=*reinterpret_cast<uint32_t*>(&hp0),p8=*reinterpret_cast<uint32_t*>(&hp8),z[2];\n  int sa=8*t+(g>>1),sb=sa+4,sh=(g&1)*16;\n  z[0]=((__shfl_sync(0xffffffff,p0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>sh)<<16);\n  z[1]=((__shfl_sync(0xffffffff,p8,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>sh)<<16);\n#pragma unroll\n  for(int pr=0;pr<2;pr++){uint32_t r[4];ldt(r,raw+4096+off(part*16+(lane&7)+8*((lane>>3)&1),h*64+pr*32+16*(lane>>4)));\n#pragma unroll\n   for(int odd=0;odd<2;odd++){uint32_t a[4];a[0]=cv2(__byte_perm(r[0],0,odd?0x4431:0x4420));a[1]=cv2(__byte_perm(r[2],0,odd?0x4431:0x4420));a[2]=cv2(__byte_perm(r[1],0,odd?0x4431:0x4420));a[3]=cv2(__byte_perm(r[3],0,odd?0x4431:0x4420));pmma(o[2*pr+odd],a,z);}}\n  __syncthreads();\n }\n asm volatile("cp.async.wait_group 0;");__syncthreads();\n float* smm=reinterpret_cast<float*>(mem);float* sml=smm+32;float* smo=sml+32;\n#pragma unroll\n for(int c=0;c<2;c++){int head=h*8+2*t+c,row=part*16+head;if(g==0){smm[row]=mm[c];sml[row]=ll[c];}\n#pragma unroll\n for(int pr=0;pr<2;pr++)for(int odd=0;odd<2;odd++){int d=pr*32+2*g+odd;smo[row*64+d]=o[2*pr+odd][c];smo[row*64+d+16]=o[2*pr+odd][c+2];}}\n __syncthreads();\n if(tid<16){float high=fmaxf(smm[tid],smm[16+tid]),ls=0.f;\n#pragma unroll\n for(int p=0;p<2;p++){float m=smm[p*16+tid],f=m==-INFINITY?0.f:exp2f(m-high);ls+=sml[p*16+tid]*f;smm[p*16+tid]=f;}\n sml[tid]=ls>0.f?1.f/ls:0.f;long row=((long)(b*2+(tid>>3))*splits+split)*16+(tid&7);LSE[row]=ls>0.f?(high+log2f(ls))*0.6931471805599453f:-INFINITY;}\n __syncthreads();\n for(int i=tid;i<1024;i+=128){int head=i>>6,d=i&63;float sum=fmaf(smo[i],smm[head],smo[1024+i]*smm[16+head]);long dst=(((long)(b*2+(head>>3))*splits+split)*16+(head&7))*64+d;PART[dst]=sum*sml[head]*vscale;}\n}\nvoid sparse_decode_run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor table,torch::Tensor seq,\n torch::Tensor ks,torch::Tensor vs,torch::Tensor part,torch::Tensor lse,int64_t page,int64_t window,\n int64_t splits,int64_t chunk,double scale,int64_t scaled){\n int dyn=3*8192+((chunk+127)/128)*128*4;cudaFuncSetAttribute(trans_decode<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,dyn);\n trans_decode<3><<<dim3(splits,seq.numel()),128,dyn,at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)q.data_ptr(),(const uint8_t*)k.data_ptr(),(const uint8_t*)v.data_ptr(),table.data_ptr<int>(),seq.data_ptr<int>(),ks.data_ptr<float>(),vs.data_ptr<float>(),part.data_ptr<float>(),lse.data_ptr<float>(),table.size(1),page,window,splits,chunk,scale,scaled);\n auto st=cudaGetLastError();TORCH_CHECK(st==cudaSuccess,cudaGetErrorString(st));}\n'

_SP_MOD = None

def _sp_module():
    global _SP_MOD
    if _SP_MOD is None:
        from torch.utils.cpp_extension import load_inline
        _SP_MOD = load_inline('muse_dense_transposed_decode_v1', _SP_CPP, cuda_sources=_SP_CUDA,
            functions=['sparse_decode_run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'], verbose=False)
    return _SP_MOD

def _sp_forward(data, q, qkv):
    cfg = data['config']
    batch = cfg['batch']
    window = cfg['window']
    # Target enough CTAs to fill H200 while keeping each chunk economical.
    splits = max(4 if window else 8, triton.next_power_of_2(triton.cdiv(256 if window else 512, batch)))
    span = window if window else data['page_table'].shape[1] * cfg['page_size']
    splits = max(splits, triton.cdiv(span, 8192))
    chunk = triton.cdiv(triton.cdiv(span, splits), 32) * 32
    partial = torch.empty((batch, 2, splits, 16, 64), dtype=torch.float32, device=qkv.device)
    lse = torch.empty((batch, 2, splits, 16), dtype=torch.float32, device=qkv.device)
    _sp_module().sparse_decode_run(q,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],
        data['k_scale'],data['v_scale'],partial,lse,cfg['page_size'],window,splits,chunk,
        cfg['sm_scale']*cfg['qk_scale_factor'],int(cfg['kv_scale'] != 1.0))
    result = torch.empty((batch,1,1024),dtype=torch.bfloat16,device=qkv.device)
    _merge_norm_gate[(batch,16)](partial,lse,qkv,result,16,2,64,1,qkv.stride(0),splits,16,num_warps=4)
    return result

_FP8P_SERIAL_CUDA='\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define HD 64\n#define WT 64              // keys per tile\n\n__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ uint32_t pack_h2(float a, float b) {\n  __half2 h = __floats2half2_rn(a, b);\n  return *reinterpret_cast<uint32_t*>(&h);\n}\n__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {\n  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;\n}\n__device__ __forceinline__ uint32_t cvt_fp8x2(uint32_t x) {   // low two e4m3 bytes -> f16x2\n  uint32_t r;\n  asm("{\\n.reg .b16 a, b;\\nmov.b32 {a, b}, %1;\\ncvt.rn.f16x2.e4m3x2 %0, a;\\n}" : "=r"(r) : "r"(x));\n  return r;\n}\n\n__device__ __forceinline__ void wg_qk(float* d, uint64_t da, uint64_t db, int accum) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %34, 0;\\n"\n               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "l"(da), "l"(db), "r"(accum));\n#endif\n}\n\n__device__ __forceinline__ void wg_pv(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n\n__device__ __forceinline__ float ex2f(float x) {\n  float y;\n  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ void wg_pv72(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n72k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n__device__ __forceinline__ void wg_pvt(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n#endif\n}\n\n\n__device__ __forceinline__ void wg_commit() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_wait0() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum);\n__device__ __forceinline__ void wg_qk128(float* d, uint64_t da, uint64_t db, int accum) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %66, 0;\\n"\n               "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\\n}\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])\n               : "l"(da), "l"(db), "r"(accum));\n#endif\n}\n__device__ __forceinline__ uint64_t mkdesc(const void* p, uint32_t lbo, uint32_t sbo) {\n  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);\n  return (uint64_t)((a >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void wg_commit_wait() {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;\\n" ::: "memory");\n#endif\n}\n__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory"); }\n\n// core-matrix layout of a [rows][64 byte] fp8 tile: 8 rows x 16 B cores, K cores 128 B apart, row groups 512 B apart\n__device__ __forceinline__ int core_off(int r, int chunk) { return (r >> 3) * 512 + chunk * 128 + (r & 7) * 16; }\n// vT: [d][key] f16, cores of 8 d-rows x 8 keys (16 B per row): key cores 128 B apart, d groups 1024 B apart\n__device__ __forceinline__ int vt_off(int d, int key) { return (d >> 3) * 1024 + (key >> 3) * 128 + (d & 7) * 16 + (key & 7) * 2; }\n\n__device__ __forceinline__ int vmn_off(int key, int d) { return (key >> 3) * 1152 + (d >> 3) * 128 + (key & 7) * 16 + (d & 7) * 2; }\n__device__ __forceinline__ void pv8_ldt(uint32_t* r,const void* p){uint32_t a=__cvta_generic_to_shared(p);asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a):"memory");}\n__device__ __forceinline__ void wg_pv8(float* c,uint32_t* a,uint64_t b){\n asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},{%32,%33,%34,%35},%36,1,1,1;"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));\n}\n\n__device__ __forceinline__ uint32_t packp8(float a,float b){uint16_t y;\n asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(y):"f"(a),"f"(b));return y;}\n\n#define WS 2   // raw stages\n#define TILEB 8192   // bytes of a raw K or V tile\n__device__ __forceinline__ void wg_qk128_nw(float* d, uint64_t da, uint64_t db, int accum) { wg_qk128(d, da, db, accum); }\n\n\n__device__ __forceinline__ void wg_qk_first(float* d, uint64_t da, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 0, 1, 1;\\n"\n               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])\n               : "l"(da), "l"(db));\n#endif\n}\n__device__ __forceinline__ void wg_qk_acc(float* d, uint64_t da, uint64_t db) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1;\\n"\n               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])\n               : "l"(da), "l"(db));\n#endif\n}\n\n#define LAZY_TH 0.0f\n__global__ void __launch_bounds__(128, 4)\nwg5_partials(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ KC, const uint8_t* __restrict__ VC,\n             const int* __restrict__ TABLE, const int* __restrict__ SEQLEN, const float* __restrict__ KS,\n             const float* __restrict__ VS, float* __restrict__ PART, float* __restrict__ LSE, int L, int HQ,\n             int table_n, int page, int window, int splits, int chunk, float sm_scale, int scaled, int dbg) {\n#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  extern __shared__ __align__(1024) uint8_t smem_raw[];\n  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);\n  const int tid = threadIdx.x, wgi = 0, tw = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;\n  const int g = lane >> 2, t = lane & 3;\n  const int split = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;\n  const int prefix = SEQLEN[b];\n  const int first = window ? max(prefix - window + 1, 0) : 0;\n  const int begin = first + split * chunk;\n  const int end = min(begin + chunk, prefix + L);\n  const float key_scale = scaled ? KS[0] : 1.f, value_scale = scaled ? VS[0] : 1.f;\n  const float sc2 = sm_scale * key_scale * 1.4426950408889634f;\n  const long base_row = ((long)(b * 2 + kh) * splits + split) * 64;\n  const int n = end > begin ? end - begin : 0;\n  const int nst = (n + WT - 1) / WT;\n\n  uint8_t* qs = smem;\n  uint8_t* vt = smem + 4096;\n  uint8_t* stages = smem + 4096 + 4096;\n  int* rows = reinterpret_cast<int*>(stages + WS * 8192);\n\n  const int pshift = __ffs(page) - 1, pmask = page - 1;\n  for (int i = tid; i < n; i += 128) {\n    const int tok = begin + i;\n    const int pg = tok >> pshift;\n    const int phys = (pg < table_n) ? TABLE[b * table_n + pg] : -1;\n    rows[i] = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));\n  }\n  for (int c = tw; c < 64 * 4; c += 128) {\n    const int r = c >> 2, ch = c & 3;\n    const int token = r >> 3, qh = kh * 8 + (r & 7);\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (token < L) v = *reinterpret_cast<const uint4*>(Q + ((long)(b * L + token) * HQ + qh) * HD + ch * 16);\n    *reinterpret_cast<uint4*>(qs + core_off(r, ch)) = v;\n  }\n  proxy_fence();\n  __syncthreads();\n\n  // ---- loop-invariant addressing ----\n  const uint32_t stages_s = (uint32_t)__cvta_generic_to_shared(stages);\n  const int ci = tid & 7;                   // 16-byte chunk within the key row (K 0-7, V 8-15)\n  const int cc = ci & 3;\n  const bool isv = ci >= 4;\n  const uint8_t* srcb = (isv ? VC : KC) + kh * 64 + cc * 16;\n  uint32_t dst_off[4];\n  int key4[4];\n#pragma unroll\n  for (int i = 0; i < 4; i++) {\n    const int key = (tid >> 3) + 16 * i;\n    key4[i] = key;\n    dst_off[i] = isv ? (4096 + key * 64 + (cc ^ (key & 3)) * 16) : core_off(key, cc);\n  }\n  auto issue = [&](int s, int stage) {\n    if (s < nst && !(dbg & 8)) {\n      const uint32_t base = stages_s + stage * 8192;\n#pragma unroll\n      for (int i = 0; i < 4; i++) {\n        const int li = s * WT + key4[i];\n        const long rr = li < n ? rows[li] : 0;\n        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"(base + dst_off[i]), "l"(srcb + rr * 128));\n      }\n    }\n    cp_commit();\n  };\n  int wr_stage = 0;\n#pragma unroll\n  for (int s = 0; s < WS - 1; s++) { issue(s, wr_stage); wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1; }\n\n  const uint64_t qd0 = mkdesc(qs, 128, 512), qd1 = mkdesc(qs + 256, 128, 512);\n  const uint64_t kd0 = mkdesc(stages, 128, 512);\n  const uint64_t vd0 = mkdesc(vt,128,512);\n  // per-thread V conversion offsets (loop invariant)\n  // rotated chunk assignment: lanes take consecutive keys but different 16 B chunks, so both the smem loads and the stores are (nearly) conflict free\n  const int vkey = (tw & 31) + 32 * (tw >> 6);\n  const int vcb = ((tw >> 5) & 1) * 2;\n  uint32_t vsrc_off[2], vdst_off[4];\n#pragma unroll\n  for (int j = 0; j < 2; j++) {\n    const int cs = vcb + j;\n    vsrc_off[j] = vkey * 64 + (cs ^ (vkey & 3)) * 16;\n    vdst_off[2 * j] = (uint32_t)vmn_off(vkey, 16 * cs);\n    vdst_off[2 * j + 1] = (uint32_t)vmn_off(vkey, 16 * cs + 8);\n  }\n\n  float o[32];\n  float ls_run[2]={0.f,0.f};\n#pragma unroll\n  for (int i = 0; i < 32; i++) o[i] = 0.f;\n  float m_run[2] = {-INFINITY, -INFINITY};\n\n  int rd_stage = 0;\n#pragma unroll 1\n  for (int s = 0; s < nst; s++) {\n    cp_wait<WS - 2>();\n    if (!(dbg & 128)) { proxy_fence(); __syncthreads(); }\n    issue(s + WS - 1, wr_stage);\n    wr_stage = wr_stage + 1 == WS ? 0 : wr_stage + 1;\n    const uint8_t* vst = stages + rd_stage * 8192 + 4096;\n    const uint64_t kdesc = kd0 + (uint64_t)(rd_stage * 512);\n    rd_stage = rd_stage + 1 == WS ? 0 : rd_stage + 1;\n\n    float sacc[32];\n    // Key-permuted FP8 transpose: P uses the identical within-block order.\n#pragma unroll\n    for(int pr=0;pr<2;pr++){\n      int key0=warp*16+(lane&7)+8*((lane>>3)&1),byte=pr*32+16*(lane>>4);\n      uint32_t r[4];pv8_ldt(r,vst+key0*64+(((byte>>4)^(key0&3))<<4));\n      uint32_t z[4]={__byte_perm(r[0],r[1],0x6420),__byte_perm(r[0],r[1],0x7531),\n                     __byte_perm(r[2],r[3],0x6420),__byte_perm(r[2],r[3],0x7531)};\n      int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);\n      uint32_t ad=__cvta_generic_to_shared(vt+core_off(d,warp));\n      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");\n    }\n    proxy_fence();\n    wg_fence();\n    wg_qk_first(sacc, qd0, kdesc);\n    wg_qk_acc(sacc, qd1, kdesc + 16);\n    wg_commit();\n    wg_wait0();\n#pragma unroll\n    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(sacc[i])::"memory");\n    asm volatile("bar.sync %0, 128;\\n" ::"r"(1 + wgi) : "memory");\n\n    const int tb = begin + s * WT;\n    const bool full_vis = (tb + WT <= end) && (L == 1 || tb + WT - 1 <= prefix) && (!window || L == 1 || tb > prefix + L - 1 - window);\n    if (!full_vis) {\n#pragma unroll\n      for (int j = 0; j < 8; j++)\n#pragma unroll\n        for (int r = 0; r < 4; r++) {\n          const int row = warp * 16 + g + 8 * (r >> 1);\n          const int token = row >> 3;\n          const int col = tb + j * 8 + 2 * t + (r & 1);\n          const int pos = prefix + token;\n          const bool vis = (token < L) && (col < prefix + L) && (col < end) && (col <= pos) && (!window || col > pos - window);\n          if (!vis) sacc[4 * j + r] = -INFINITY;\n        }\n    }\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int j = 0; j < ((dbg & 64) ? 0 : 8); j++) {\n      mx0 = fmaxf(mx0, fmaxf(sacc[4 * j], sacc[4 * j + 1]));\n      mx1 = fmaxf(mx1, fmaxf(sacc[4 * j + 2], sacc[4 * j + 3]));\n    }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));\n    mx0 *= sc2; mx1 *= sc2;\n    // lazy rescale: the reference max only moves when a tile exceeds it by more than LAZY_TH (log2 domain), so p <= 2^LAZY_TH\n    if (__any_sync(0xffffffffu, mx0 > m_run[0] + LAZY_TH || mx1 > m_run[1] + LAZY_TH)) {\n      const float n0 = fmaxf(m_run[0], mx0), n1 = fmaxf(m_run[1], mx1);\n      const float s0 = n0 == -INFINITY ? 0.f : n0, s1 = n1 == -INFINITY ? 0.f : n1;\n      const float c0 = ex2f(m_run[0] - s0), c1 = ex2f(m_run[1] - s1);\n      m_run[0] = n0; m_run[1] = n1;ls_run[0]*=c0;ls_run[1]*=c1;\n#pragma unroll\n      for (int j = 0; j < 8; j++) {\n        o[4 * j] *= c0; o[4 * j + 1] *= c0;\n        o[4 * j + 2] *= c1; o[4 * j + 3] *= c1;\n      }\n    }\n    const float sf0 = m_run[0] == -INFINITY ? 0.f : m_run[0], sf1 = m_run[1] == -INFINITY ? 0.f : m_run[1];\n    uint32_t pa[8][2];\n#pragma unroll\n    for (int j = 0; j < 8; j++) {\n      const float p0 = (dbg & 32) ? sacc[4 * j] : ex2f(fmaf(sacc[4 * j], sc2, 8.f-sf0)), p1 = (dbg & 32) ? sacc[4 * j + 1] : ex2f(fmaf(sacc[4 * j + 1], sc2, 8.f-sf0));\n      const float p2 = (dbg & 32) ? sacc[4 * j + 2] : ex2f(fmaf(sacc[4 * j + 2], sc2, 8.f-sf1)), p3 = (dbg & 32) ? sacc[4 * j + 3] : ex2f(fmaf(sacc[4 * j + 3], sc2, 8.f-sf1));\n      ls_run[0]+=p0+p1;ls_run[1]+=p2+p3;\n      pa[j][0] = packp8(p0,p1);\n      pa[j][1] = packp8(p2,p3);\n    }\n    uint32_t pp[2][4];\n#pragma unroll\n    for(int kk=0;kk<2;kk++){\n#pragma unroll\n      for(int h=0;h<2;h++){\n        pp[kk][h]=pa[kk*4][h]|(pa[kk*4+1][h]<<16);\n        pp[kk][h+2]=pa[kk*4+2][h]|(pa[kk*4+3][h]<<16);\n      }\n    }\n    wg_fence();wg_pv8(o,pp[0],vd0);wg_pv8(o,pp[1],vd0+16);\n    wg_commit_wait();\n#pragma unroll\n    for (int i = 0; i < 32; i++) asm volatile("" : "+f"(o[i])::"memory");\n  }\n  cp_wait<0>();\n#pragma unroll\n  for (int h = 0; h < 2; h++) {\n    const int row = warp * 16 + g + 8 * h;\n    float ls=ls_run[h];ls+=__shfl_xor_sync(0xffffffff,ls,1);ls+=__shfl_xor_sync(0xffffffff,ls,2);\n    const float inv = ls > 0.f ? 1.f / ls : 1.f;\n#pragma unroll\n    for (int j = 0; j < 8; j++) {\n      PART[(base_row + row) * HD + j * 8 + 2 * t] = o[4 * j + 2 * h] * inv * value_scale;\n      PART[(base_row + row) * HD + j * 8 + 2 * t + 1] = o[4 * j + 2 * h + 1] * inv * value_scale;\n    }\n    if (t == 0) LSE[base_row + row] = ls > 0.f ? (m_run[h] + log2f(ls) - 8.f) * 0.6931471805599453f : -INFINITY;\n  }\n#endif\n}\n\nextern "C" int wg5_run(const void* Q, const void* KC, const void* VC, const int* TABLE, const int* SEQLEN, const float* KS,\n                       const float* VS, float* PART, float* LSE, int B, int L, int HQ, int table_n, int page,\n                       int window, int splits, int chunk, float sm_scale, int scaled, int dbg, cudaStream_t st) {\n  const size_t rmax = ((chunk + 255) / 256) * 256;\n  const size_t dyn = 1024 + 4096 + 4096 + (size_t)WS * 8192 + rmax * 4;\n  cudaFuncSetAttribute(wg5_partials, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);\n  dim3 grid(splits, 2, B);\n  wg5_partials<<<grid, 128, dyn, st>>>(reinterpret_cast<const uint8_t*>(Q), reinterpret_cast<const uint8_t*>(KC),\n                                       reinterpret_cast<const uint8_t*>(VC), TABLE, SEQLEN, KS, VS, PART, LSE, L, HQ,\n                                       table_n, page, window, splits, chunk, sm_scale, scaled, dbg);\n  return 0;\n}\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\nextern "C" int wg5_run(const void*, const void*, const void*, const int*, const int*, const float*, const float*, float*, float*,\n                       int, int, int, int, int, int, int, int, float, int, int, cudaStream_t);\nvoid wg5_run_t(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor table, torch::Tensor seqlens, torch::Tensor ks,\n               torch::Tensor vs, torch::Tensor part, torch::Tensor lse, int64_t L, int64_t HQ, int64_t page,\n               int64_t window, int64_t splits, int64_t chunk, double sm_scale, int64_t scaled, int64_t dbg) {\n  int r = wg5_run(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), table.data_ptr<int>(), seqlens.data_ptr<int>(), ks.data_ptr<float>(),\n                  vs.data_ptr<float>(), part.data_ptr<float>(), lse.data_ptr<float>(), (int)seqlens.numel(), (int)L, (int)HQ,\n                  (int)table.size(1), (int)page, (int)window, (int)splits, (int)chunk, (float)sm_scale, (int)scaled, (int)dbg,\n                  at::cuda::getCurrentCUDAStream());\n  TORCH_CHECK(r == 0, "wg5");\n}\n'
