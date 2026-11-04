"""decoder_attn1_v1: put the attention board's best kernel (attn_b1full_v1/submission_stack.py, judge 2.697/2.733x)
into the decoder parent (decoder_stack_v1/submission.py). Same launch modes and plan; adds the decoder's pooler L2
prefetch (pf) and its run() signature. Variant C: also the decoder's early PDL trigger right after pdl_wait."""
import re, sys, os
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, '..', '..')
par = open(os.path.join(R, 'experiments/decoder_stack_v1/submission.py')).read()
brd = open(os.path.join(R, 'experiments/attn_b1full_v1/submission_stack.py')).read()

def body(src, start_tag):
    i = src.index(start_tag) + len(start_tag)
    j = src.index('\n"""', i)
    return i, j

def rep1(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:80])
    return s.replace(a, b)

bi, bj = body(brd, 'CUDA_SRC = r"""')
cu = brd[bi:bj]
cu = rep1(cu, "  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n};",
          "  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n"
          "  const uint8_t* pf;     // decode layer: next projection's weights, prefetched into L2 during attention\n"
          "  unsigned pf_chunk;    // bytes per CTA (0: no prefetch)\n};")
cu = rep1(cu, "  const int pair = blockIdx.z;\n  if (!FLAT) {\n",
          "  const int pair = blockIdx.z;\n"
          "  if (p.pf_chunk && threadIdx.x == 0) {\n"
          "    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;\n"
          "    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {\n"
          "      const unsigned n = min(32768u, p.pf_chunk - o);\n"
          "      asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(src + o), \"r\"(n) : \"memory\");\n"
          "    }\n  }\n  if (!FLAT) {\n")
cu = rep1(cu, "std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,\n         int64_t tma) {",
          "std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,\n"
          "         int64_t tma, int64_t pf_ptr, int64_t pf_bytes) {\n"
          "  (void)scratch; (void)bar;   // (the decoder's sm_80 cluster-emulation buffers: unused by this sm_90a build)")
cu = rep1(cu, "  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));\n",
          "  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));\n"
          "  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);\n"
          "  p.pf_chunk = pf_ptr ? (unsigned)((pf_bytes / ((int64_t)grid.x * grid.y * grid.z)) & ~15ll) : 0u;\n")
variant = sys.argv[1] if len(sys.argv) > 1 else 'B'
if variant == 'C':
    cu = rep1(cu, "  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it\n",
              "  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it\n"
              "  pdl_launch_dependents();   // (decode layer: the next kernel's prologue overlaps this one)\n")
pi, pj = body(par, 'ATT7_CUDA_SRC = r"""')
out = par[:pi] + cu + par[pj:]
out = rep1(out, 'name=f"muse_att_v298w1pf_{arch}"', 'name=f"muse_att_b1fstack%s_{arch}"' % variant)
out = rep1(out, 'extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"]',
           'extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"]')
out = rep1(out, "    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)\n}",
           "    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)\n"
           "    # (attention board, attn_slide_v1 r6: with bf16 q_len-1 partials, b16 q1 sliding at 16 splits beats 8 by ~9%)\n"
           "    (16, 1, True, 1): dict(n_splits=16, coop=0, nc=1),\n}")
dst = os.path.join(H, 'submission.py' if variant == 'B' else f'var{variant}.py')
open(dst, 'w').write(out)
print(dst, len(out.encode()))
