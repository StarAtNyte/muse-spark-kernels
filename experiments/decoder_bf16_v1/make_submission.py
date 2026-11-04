"""Build decoder_bf16_v1/submission.py from the readable parent (decoder_readable_fp32_ybuf_v4).

(1) re-enable the rs128 BF16 split-reduction backend (readable _make_chen block, chenwainuo muse_bf16_proj, Apache-2.0,
    as carried in decode_keell0renz_compare_v1; dead GX/GE/GRD/router-variant tables and unused RQV entries dropped);
(2) restore the BF16 WGS ybuf and the BF16-output torch fallbacks;
(3) apply the g16 attention linear-exp2 hunks (decoder_g16_v1 vs parent), then g16b (attn_vfull_v1 g16 -> g16b).
"""
import re, subprocess, sys, pathlib
E = pathlib.Path(__file__).resolve().parent.parent
par = (E / "decoder_readable_fp32_ybuf_v4/submission.py").read_text()
ref = (E / "decode_keell0renz_compare_v1/submission.py").read_text().split("\n")

def once(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a))
    return s.replace(a, b)

# (3) g16 port: patch the parent with the g16 diff
tmp = pathlib.Path("/tmp/claude-1000/bf16_build.py")
tmp.write_text(par)
d = subprocess.run(["diff", str(E / "decoder_readable_fp32_ybuf_v4/submission.py"), str(E / "decoder_g16_v1/submission.py")],
                   capture_output=True, text=True).stdout
subprocess.run(["patch", "-s", str(tmp)], input=d, text=True, check=True)
s = tmp.read_text()

# (3b) g16b (attention check 979f4c2f587f4325): linear exp2 in the per-request q_len-8 pack too, and BF16x2 split-KV
# partials (red.global.add.noftz.v2.bf16x2) in the per-request q_len-8 kernels only (L == 8 && !FLAT). The 15 hunks that
# match apply as a patch; the 6 that do not (the decoder's attention carries extra OKC_SC1 / FINWARM / RAWTHR paths)
# are applied by hand below.
gd = subprocess.run(["diff", "-u", str(E / "attn_vfull_v1/submission_g16.py"), str(E / "attn_vfull_v1/submission_g16b.py")],
                    capture_output=True, text=True).stdout
r = subprocess.run(["patch", "-s", "-r", "/dev/null", str(tmp)], input=gd, text=True, capture_output=True)
assert "6 out of 21 hunks FAILED" in r.stdout, r.stdout
s = tmp.read_text()
assert "acc_red4" in s
s = once(s, "template <int L>\n__device__ __forceinline__ void sum_copies(", "template <int L, bool BF>\n__device__ __forceinline__ void sum_copies(")
s = once(s, """      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);""", """      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = acc_ld4<BF>(p, acc, 0);
      const float4 xb = acc_ld4<BF>(p, acc, 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      acc_zero4<BF>(p, acc, 0);
      acc_zero4<BF>(p, acc, 1);""")
s = once(s, "template <int L, int NT>\n__device__ __forceinline__ void fin_nc1(", "template <int L, int NT, bool BF>\n__device__ __forceinline__ void fin_nc1(")
s = once(s, "fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);", "fin_nc1<L, NT, BFP>(p, b, pair, bp, 0.f, 1);")
s = once(s, "      if constexpr (PCHK) { dlo = lin_snap(dlo); dhi = lin_snap(dhi); cr[0] = dlo; cr[1] = dhi; }\n",
         "      dlo = lin_snap(dlo); dhi = lin_snap(dhi); cr[0] = dlo; cr[1] = dhi;   // (g16b: every q_len 8 pack is linear)\n")
s = once(s, "const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);", "const float nlo = lin_snap(fmaxf(mlo, cr[0])), nhi = lin_snap(fmaxf(mhi, cr[1]));")
s = once(s, """        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));""",
"""        {   // linear exp2 straight to e4m3 bytes (as the flat q_len 8 pack)
          const __half2 blo = lin_bias(dlo), bhi = lin_bias(dhi);
          pa[kk][0] = lin4h(sv[0], sv[1], sv[4], sv[5], lin_a, blo);
          pa[kk][1] = lin4h(sv[2], sv[3], sv[6], sv[7], lin_a, bhi);
          pa[kk][2] = lin4h(sv[8], sv[9], sv[12], sv[13], lin_a, blo);
          pa[kk][3] = lin4h(sv[10], sv[11], sv[14], sv[15], lin_a, bhi);
        }""")
s = once(s, "              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);",
         "              const float nlo = lin_snap(fmaxf(hlo, cr[0])), nhi = lin_snap(fmaxf(hhi, cr[1]));   // (bytes use the snapped offset)")
assert "EXQ(fmaf" not in s and "fin_nc1<L, NT>(" not in s and "sum_copies<L>(" not in s

# (2) BF16 ybuf + BF16 fallbacks (as in the board-best lineage)
s = once(s, "#define OKC_YBF 0   // keep per-route values FP32 until final output", "#define OKC_YBF 1   // ybf: WGS ybuf in bf16")
s = once(s, 'return torch.mm(x, state["raw"][layer][:n].t(), out_dtype=torch.float32).to(torch.bfloat16)  # FP32 until final output',
         'return F.linear(x, state["raw"][layer][:n])   # what SGLang\'s bf16 dispatch runs on H200')
s = once(s, 'qkv = torch.mm(data["attn_input"], state["qkv"].t(), out_dtype=torch.float32).to(torch.bfloat16)',
         'qkv = torch.mm(data["attn_input"], state["qkv"].t())')
s = once(s, 'pooled = torch.mm(h, state["pr"][:D_EXPERT].t(), out_dtype=torch.float32).to(torch.bfloat16)',
         'pooled = F.linear(h, state["pr"][:D_EXPERT])')

# (1) rs128: locate the readable block in the reference
i0 = next(i for i, l in enumerate(ref) if l.startswith("def _make_chen():"))
i1 = next(i for i, l in enumerate(ref) if l.startswith("def _rq_gemm(state, name, x, T):"))
i2 = i1 + 1
while not ref[i2].startswith("# ----"):
    i2 += 1
blk = ref[i0:i2]
# drop dead tables (never referenced by the rq path)
def cut(lines, start_pred, end_pred):
    a = next(i for i, l in enumerate(lines) if start_pred(l))
    b = next(i for i in range(a, len(lines)) if end_pred(lines[i]))
    return lines[:a] + lines[b:]
blk = cut(blk, lambda l: l.startswith("    GXP = []"), lambda l: l.startswith("    _OS = ("))
blk = cut(blk, lambda l: l.startswith("    RWV = {100:"), lambda l: l.startswith("    RQV = ["))
# keep only RQV entries that are built/baked/referenced
keep = set(re.findall(r'"([A-Za-z0-9]+)"', "\n".join(l for l in blk if l.startswith("    RQPROD = [") or l.startswith("              \"d2s5hzwP\"")))) 
src = "\n".join(blk)
m = list(re.finditer(r"    RQPROD = \[([^\]]*)\]", src))[-1]
keep = set(re.findall(r'"([^"]+)"', m.group(1)))
out = []
inv = False
for l in blk:
    if l.startswith("    RQV = ["):
        inv = True
    elif inv and l.startswith("    ]"):
        inv = False
    elif inv:
        mm = re.match(r'\s+\("([^"]+)"', l)
        if mm and mm.group(1) not in keep:
            continue
    out.append(l)
blk = out
src = "\n".join(blk)
# RQCUR as a name set (its variants need not all be built)
src = re.sub(r"RQCUR = \{RQI\[n\] for n in \(", "RQCUR = set((", src)
src = src.replace(")}  # variants that need the FL bit 20 cursor array", "))  # variants that need the FL bit 20 cursor array")
src = once(src, "if esc2 is None and RQI[nm] in RQCUR:", "if esc2 is None and nm in RQCUR:")
assert "GXP" not in src and "RWV" not in src
hdr = ("# ---- rs128 BF16 split-reduction GEMV for the T=128/256 qkv and T=128 unpooler projections (re-enabled: the organizer\n"
       "# clarification of 2026-11-03/04 allows BF16 partials and a BF16 combine within tolerance). Source: chenwainuo/okc\n"
       "# muse_bf16_proj @ 3d418e1 (Apache-2.0), via team keell0renz; trimmed to the rs128 path.\n")
stub = ("# The BF16 split-reduction backend is disabled pending an FP32 replacement.\nimport time as _rq_time\n\n"
        "def _rq_gemm(state, name, x, T):\n    return None\n")
s = once(s, stub, hdr + src + "\n")
dst = E / "decoder_bf16_v1/submission.py"
dst.write_text(s)
import ast; ast.parse(s)
print(len(s.encode()), "bytes; RQV kept", sorted(keep))
