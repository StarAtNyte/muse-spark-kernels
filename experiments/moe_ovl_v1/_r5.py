"""moe_big_v1 shared driver helpers (inlined into each driver by run_modal: the driver execs this file's text).
Patches the MAIN (WGS) CUDA source: overridable cost tables, NOAB ablation, light DBG (gtimer stamps + per-CTA stage
histograms by n-tiles + doneAs spin cycles)."""
import importlib.util, json, math, re, statistics, sys, time, concurrent.futures
import torch
T0 = time.time()
def el(): return round(time.time() - T0, 1)
sys.path.insert(0, "/w/kernels/muse_moe_w4a16")
import reference as R

def load_sub(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

def rep(s, old, new, n=1):
    c = s.count(old); assert c == n, (c, old[:80]); return s.replace(old, new)

def patch_common(a):
    a = rep(a, "#define OKC_DEBUG 0", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n#ifndef OKC_NOAB\n#define OKC_NOAB 0\n#endif")
    # light DBG: doneAs spin cycles (x producer) into ext slot 30, stage histogram per CTA in wgs_ranges
    old = "if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }"
    new = ("if (pb && !(OKC_WGSX & 8)) { const long long c0_ = OKC_DEBUG ? clock64() : 0; while (!OKC_NOAB && ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32);"
           " if (OKC_DEBUG && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 30] += clock64() - c0_; }")
    a = rep(a, old, new)
    old = "  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}"
    new = ("  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n#if OKC_DEBUG >= 2\n  if (p.dbg) { unsigned long long* e_ = p.dbg + gridDim.x * 8 + blockIdx.x * 32;\n"
           "    for (int pb = 0; pb < 2; ++pb) { int lu = -1; for (int s = S.wgr[2 * pb]; s < S.wgr[2 * pb + 1]; ++s) { const int j = s >> (9 - pb); int nt = (S.jC[j] + 7) >> 3; nt = nt > 8 ? 8 : nt;"
           " e_[pb * 12 + nt - 1] += 1; if ((s >> 4) != lu) { e_[pb * 12 + 8] += 1; lu = s >> 4; } } }\n  }\n#endif\n}")
    a = rep(a, old, new)
    return a

def set_tables(a, A=None, B=None, U=None, UB=None):
    if A is not None: a = re.sub(r"#define OKC_WGSA [^\n]*", "#define OKC_WGSA " + ", ".join(map(str, A)), a, count=1)
    if B is not None: a = re.sub(r"#define OKC_WGSB [^\n]*", "#define OKC_WGSB " + ", ".join(map(str, B)), a, count=1)
    if U is not None: a = re.sub(r"#define OKC_WGSU 80[^\n]*", f"#define OKC_WGSU {U}", a, count=1)
    if UB is not None: a = re.sub(r"#define OKC_WGSUB 70[^\n]*", f"#define OKC_WGSUB {UB}", a, count=1)
    return a

def route_stats(d, lo=0, cnt=32, sk=8, ft=64):
    ids = d["topk_ids"].long().flatten(); ids = ids[(ids >= lo) & (ids < lo + cnt)] - lo
    c = torch.bincount(ids, minlength=cnt).tolist()
    D = sum(1 for v in c if v > 0); routes = sum(c)
    fat = [v for v in c if v > sk]; dup = sum((v + ft - 1) // ft - 1 for v in fat)
    jobs = sum((v + ft - 1) // ft for v in fat) + sum(1 for v in c if 0 < v <= sk)
    return dict(D=D, routes=routes, maxc=max(c), fat=len(fat), jobs=jobs, dup=dup, counts=sorted([v for v in c if v], reverse=True))

def floors(st):
    E = 28311552
    bw = (st["D"]) * E / 4.2e12 * 1e6
    bwdup = (st["D"] + st["dup"]) * E / 4.2e12 * 1e6
    # compute: padded tokens (to 8 per job) x 2*3*4096^2 at 600 TFLOP/s
    pad = 0
    for v in st["counts"]:
        if v <= 8: pad += 8
        else:
            q = v
            while q > 0: pad += (min(q, 64) + 7) // 8 * 8; q -= 64
    comp = st["routes"] * 2 * 3 * 4096 ** 2 / 600e12 * 1e6
    comp_pad = pad * 2 * 3 * 4096 ** 2 / 600e12 * 1e6
    return dict(bw42=round(bw, 1), bw42_dup=round(bwdup, 1), comp600=round(comp, 1), comp600_pad=round(comp_pad, 1))

class Bench:
    def __init__(self, S, w):
        self.S = S; self.w = w
        lo, cnt = S.LOCAL_START, S.LOCAL_EXPERTS
        prmt = S.PRMT_SCALES and S._scales_prmt_exact(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
        sc6 = bool(prmt) and S._sc6_ok(w["w13_sf"][lo:lo + cnt], w["w2_sf"][lo:lo + cnt])
        sfx = ("-DOKC_PRMTSC=1",) if prmt else ()
        self.sfxm = sfx + (("-DOKC_SC6W=1",) if (sc6 and S.SC6_WGS) else ())
        self.MAIN = tuple(S._main_defines(sfx)) + self.sfxm
        self.flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    def setup(self, st):
        self.st = st; self.G = st["grid"]
        eg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(eg): pass
        self.empty = statistics.median([self.t1(eg) for _ in range(30)])
    def launcher(self, defs, d, out, sc=None):
        ext = self.S._ext(defs); st = self.st; G = self.G
        x = d["hidden_states"].contiguous(); ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
        return lambda: ext.okc_run(st["w"], sc or st["scratch"], x, ids, tw, out, 0, 32, G)
    def capture(self, fn):
        fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        g.replay(); torch.cuda.synchronize(); return g
    def t1(self, g):
        self.flush.sum(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1000
    def check(self, defs, d):
        out = torch.empty((d["hidden_states"].shape[0], 4096), dtype=torch.bfloat16, device="cuda")
        self.launcher(defs, d, out)(); torch.cuda.synchronize()
        return R.check_implementation(d, out, self.w)
    def paired(self, arms, d, reps=16):
        bs = d["hidden_states"].shape[0]
        gs = {n: self.capture(self.launcher(df, d, torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda"))) for n, df in arms.items()}
        names = list(arms); T = {n: [] for n in names}
        for r in range(reps):
            order = names[r % len(names):] + names[:r % len(names)]
            if r & 1: order = order[::-1]
            for n in order: T[n].append(self.t1(gs[n]) - self.empty)
        del gs
        row = {n: round(statistics.median(v), 2) for n, v in T.items()}
        for n in names[1:]: row["r_" + n] = round(statistics.median(sorted(b / c for b, c in zip(T[names[0]], T[n]))), 4)
        return row
    def timeline(self, defs, d, reps=6):
        G = self.G; bs = d["hidden_states"].shape[0]
        out = torch.empty((bs, 4096), dtype=torch.bfloat16, device="cuda")
        dbg = torch.zeros(G * 8 + G * 32, dtype=torch.int64, device="cuda")
        fn = self.launcher(defs, d, out, list(self.st["scratch"])[:3] + [dbg])
        res = []
        for r in range(reps):
            self.flush.sum(); dbg.zero_(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
            if r < 2: continue
            tt = dbg[:G * 8].view(G, 8).cpu().double(); ex = dbg[G * 8:].view(G, 32).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            res.append(((tt - t00) / 1000, ex))
        return res

def patch_early_w(a):
    """W producer issues its first WGS_NS stages as soon as the x producer has the stage ranges (bar 8), i.e. during
    the consumers' x gather, and joins bar 2 only before its first empty-slot wait."""
    old = ("  asm volatile(\"bar.sync 7, %0;\\n\" ::\"n\"(NT + 32) : \"memory\");\n  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n  __syncwarp();\n")
    new = old + "#if OKC_EARLYW\n  asm volatile(\"bar.arrive 8, 64;\\n\" ::: \"memory\");\n#endif\n"
    a = rep(a, old, new)
    old = ("__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n  WS_JOIN();  // routing tables and this CTA's stage ranges are ready\n"
           "  if ((threadIdx.x & 31) != 0) return;\n  int ws = 0, n = 0;\n")
    new = ("__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n#if OKC_EARLYW\n  asm volatile(\"bar.sync 8, 64;\\n\" ::: \"memory\");\n"
           "  bool joined_ = false;\n  const bool l0_ = (threadIdx.x & 31) == 0;\n#else\n  WS_JOIN();  // routing tables and this CTA's stage ranges are ready\n"
           "  if ((threadIdx.x & 31) != 0) return;\n  constexpr bool l0_ = true;\n#endif\n  int ws = 0, n = 0;\n")
    a = rep(a, old, new)
    old = ("      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n")
    new = ("#if OKC_EARLYW\n      if (n == WGS_NS && !joined_) { __syncwarp(); WS_JOIN(); joined_ = true; if (!l0_) return; }\n#endif\n" + old)
    a = rep(a, old, new)
    # the issue itself: lane 0 only (all lanes run the loop until the join)
    old = ("      uint64_t* bar = S.bars + BW_F + ws;\n      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;\n"
           "      asm volatile(\"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n\" ::\"r\"(smem_u32(bar)), \"r\"(WGS_STG) : \"memory\");\n")
    new = ("      uint64_t* bar = S.bars + BW_F + ws;\n      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;\n"
           "      if (l0_) {\n      asm volatile(\"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n\" ::\"r\"(smem_u32(bar)), \"r\"(WGS_STG) : \"memory\");\n")
    a = rep(a, old, new)
    old = ("                   ::\"r\"(smem_u32(ring + ws * UNIT_BYTES)), \"l\"(src), \"r\"(WGS_STG), \"r\"(smem_u32(bar)) L2H_ARG : \"memory\");\n"
           "      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n}")
    new = ("                   ::\"r\"(smem_u32(ring + ws * UNIT_BYTES)), \"l\"(src), \"r\"(WGS_STG), \"r\"(smem_u32(bar)) L2H_ARG : \"memory\");\n      }\n"
           "      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n#if OKC_EARLYW\n  if (!joined_) { __syncwarp(); WS_JOIN(); }\n#endif\n}")
    a = rep(a, old, new)
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_EARLYW\n#define OKC_EARLYW 0\n#endif\n")
    return a

def patch_abl(a):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_ABLW\n#define OKC_ABLW 0\n#endif\n#ifndef OKC_NOFIN\n#define OKC_NOFIN 0\n#endif\n#ifndef OKC_NOGATH\n#define OKC_NOGATH 0\n#endif\n")
    old = "const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;"
    new = ("const unsigned char* src = (OKC_ABLW == 2 || (OKC_ABLW == 1 && j > 0 && S.jE[j - 1] == S.jE[j])) ? base + (size_t)(blockIdx.x * 2 + (st & 1)) * WGS_STG"
           " : base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;")
    a = rep(a, old, new)
    a = rep(a, "    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n#if OKC_MFIN && (OKC_MFP & 4)", "    for (int tk = blockIdx.x; tk < (OKC_NOFIN ? 0 : p.M); tk += gridDim.x) {\n#if OKC_MFIN && (OKC_MFP & 4)")
    a = rep(a, "        if (r < cnt) {\n          const int ntt = (cnt + 7) >> 3;", "        if (r < cnt && !OKC_NOGATH) {\n          const int ntt = (cnt + 7) >> 3;")
    return a

def patch_pipe(a):
    """wgs_seg pipeline knobs per n-tile count: GS = 4 k16 steps per commit group for NTT <= OKC_GS4N; wgmma wait depth
    OKC_WDV (4 register sets) for NTT <= OKC_WD2N. Defaults reproduce the base."""
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_GS4N\n#define OKC_GS4N 0\n#endif\n#ifndef OKC_WD2N\n#define OKC_WD2N 0\n#endif\n#ifndef OKC_WDV\n#define OKC_WDV 2\n#endif\n")
    a = rep(a, "  constexpr int GS = OKC_WGSGS;\n  uint32_t af[2][GS][4];",
            "  constexpr int GS = (NTT <= OKC_GS4N) ? 4 : OKC_WGSGS;\n  constexpr int WD = (NTT <= OKC_WD2N) ? OKC_WDV : 1;\n  constexpr int NB = WD > 1 ? 4 : 2;\n"
            "  constexpr int NGR = 16 / GS;\n  constexpr int FCI = OKC_WGSFCI < NGR ? OKC_WGSFCI : NGR - 1;\n  uint32_t af[NB][GS][4];")
    a = rep(a, "af[ci & 1]", "af[ci % NB]", 8)
    a = rep(a, "        wg_commit();\n        wg_wait<1>();", "        wg_commit();\n        wg_wait<WD>();")
    a = rep(a, "        if (ci == 0) {\n          if (prev >= 0)", "        if (ci == WD - 1) {\n          if (prev >= 0)")
    a = rep(a, "if (FIRST && ci == OKC_WGSFCI && pd.kind)", "if (FIRST && ci == FCI && pd.kind)")
    return a

def patch_fin(a):
    """OKC_YBFX=1: BF16 per-route y (phase-B epilogue + final reads; organizer 2026-11-04 allows BF16 per-expert storage);
    OKC_FHOLD=1: the final pass keeps each route's fp32 y row slice in registers between its two passes (no re-read)."""
    a = a.replace("#define OKC_YBF 0   // preserve FP32 per-expert output for final combination", "#define OKC_YBF OKC_YBFX", 1)
    a = rep(a, "#else\n#define OKC_YBF 0\n#endif", "#else\n#define OKC_YBF OKC_YBFX\n#endif")
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_YBFX\n#define OKC_YBFX 0\n#endif\n#ifndef OKC_FHOLD\n#define OKC_FHOLD 0\n#endif\n")
    L = "          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n#endif\n"
    a = rep(a, "      float ps[8];\n", "      float ps[8];\n#if OKC_FHOLD\n      float4 ya_[8], yb_[8];\n#endif\n")
    a = rep(a, L + "          ps[sl] = ", L + "#if OKC_FHOLD\n          ya_[sl] = a; yb_[sl] = b;\n#endif\n          ps[sl] = ")
    a = rep(a, L + "          acc[0]", L.replace("          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n",
            "#if OKC_FHOLD\n          const float4 a = ya_[sl], b = yb_[sl]; (void)src;\n#else\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n#endif\n") + "          acc[0]")
    return a

def patch_dq(a):
    """ablations (wrong output): OKC_NODQ = 1-op decode; OKC_NOSCM = no per-weight scale multiply in wgs_seg"""
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_NODQ\n#define OKC_NODQ 0\n#endif\n#ifndef OKC_NOSCM\n#define OKC_NOSCM 0\n#endif\n")
    a = rep(a, "constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;",
            "constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n#if OKC_NODQ\n  r1 = r0 ^ 0x00400040u; r2 = r0 ^ 0x00800080u; r3 = r0 ^ 0x01000100u; return;\n#endif")
    old = ("          af[ci % NB][jj][0] = bmul2l(r0, sgp); af[ci % NB][jj][1] = bmul2h(r1, sgp);\n"
           "          af[ci % NB][jj][2] = bmul2l(r2, sgp); af[ci % NB][jj][3] = bmul2h(r3, sgp);\n")
    new = ("#if OKC_NOSCM\n          af[ci % NB][jj][0] = r0 ^ sgp; af[ci % NB][jj][1] = r1; af[ci % NB][jj][2] = r2; af[ci % NB][jj][3] = r3;\n#else\n" + old + "#endif\n")
    return rep(a, old, new)

# ---- moe_ovl_v1 patches on the WGS MAIN consumer (wgs_seg stage lambda)
# OKC_DAN: dual accumulators for NTT <= OKC_DAN (commit group ci accumulates into d (ci even) or d2 (ci odd); the two
#          groups in flight are independent, so the tensor pipe is not serialized on one accumulator chain)
# OKC_ERN: early weight-slot release for NTT <= OKC_ERN (all 4 code groups loaded at stage start, slot freed at once)
def patch_ovl(a):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_DAN\n#define OKC_DAN 0\n#endif\n#ifndef OKC_ERN\n#define OKC_ERN 0\n#endif\n")
    a = rep(a, "  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma",
            "  constexpr bool DA = NTT <= OKC_DAN, ER = NTT <= OKC_ERN;\n  float d2[WGS_NT][4];\n  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma")
    old = "    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    // KPERM3"
    new = ("    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    uint4 cwe1, cwe2, cwe3;\n"
           "    if constexpr (ER) {\n      cwe1 = *reinterpret_cast<const uint4*>(pc + ((4 + wp) * 32 + lane) * 16);\n"
           "      cwe2 = *reinterpret_cast<const uint4*>(pc + ((8 + wp) * 32 + lane) * 16);\n"
           "      cwe3 = *reinterpret_cast<const uint4*>(pc + ((12 + wp) * 32 + lane) * 16);\n    }\n    // KPERM3")
    a = rep(a, old, new)
    old = ("    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);\n#endif\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n"
           "      const uint4 cwg = cwn;\n")
    new = ("    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);\n#endif\n"
           "    if constexpr (ER) { __syncwarp(); if (lane == 0) mbar_arrive1(S.bars + BW_E + ws); }\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n"
           "      const uint4 cwg = (!ER || lg == 0) ? cwn : lg == 1 ? cwe1 : lg == 2 ? cwe2 : cwe3;\n")
    a = rep(a, old, new)
    old = ("      if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n      } else if (lane == 0) {\n")
    new = ("      if (ER) {\n      } else if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n      } else if (lane == 0) {\n")
    a = rep(a, old, new)
    old = ("          if (FIRST && ci == 0 && jj == 0)\n            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\n          else\n"
           "            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n"
           "                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n")
    new = ("          if (DA && (ci & 1)) {\n            if (FIRST && ci == 1 && jj == 0)\n"
           "              WGSMMA<NTT>::mma0(d2, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3], xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n"
           "            else\n              WGSMMA<NTT>::mma(d2, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3], xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n"
           "          } else\n" + old)
    a = rep(a, old, new)
    old = "  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8)"
    new = ("  wg_wait<0>();\n  if constexpr (DA) {\n#pragma unroll\n    for (int nt = 0; nt < NTT; ++nt) { d[nt][0] += d2[nt][0]; d[nt][1] += d2[nt][1]; d[nt][2] += d2[nt][2]; d[nt][3] += d2[nt][3]; }\n  }\n"
           "  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8)")
    a = rep(a, old, new)
    return a

# OKC_FIN1 (needs OKC_YBF): the phase-B epilogue writes each route's sum of squares as 256 plain per-warp partials
# (rows of a 16-row group, shuffled over the 8 row lanes; no atomics) into the free upper half of ybuf; the final pass
# then reads every route's 256 partials and its BF16 y row together: one L2 round trip and one barrier per token
def patch_fin1(a):
    a = rep(a, "#define OKC_DEBUG 0\n#endif\n", "#define OKC_DEBUG 0\n#endif\n#ifndef OKC_FIN1\n#define OKC_FIN1 0\n#endif\n")
    old = ("        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n#endif\n      }\n    }\n")
    new = old + ("#if OKC_FIN1 & 1\n    {\n      float* ssb = p.ybuf + (size_t)EMAX * HD / 2;\n      float v[2];\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n"
                 "        v[h] = 0.f;\n        if (tok0 + h < cnt) {\n          const float a0 = __bfloat162float(__float2bfloat16(d4[h] * w0)), a1 = __bfloat162float(__float2bfloat16(d4[2 + h] * w1));\n"
                 "          v[h] = a0 * a0 + a1 * a1;\n        }\n#pragma unroll\n        for (int o = 4; o < 32; o <<= 1) v[h] += __shfl_xor_sync(0xffffffffu, v[h], o);\n      }\n"
                 "      if (g == 0) {\n#pragma unroll\n        for (int h = 0; h < 2; ++h)\n          if (tok0 + h < cnt) ssb[(size_t)(jst + tok0 + h) * 256 + r16] = v[h];\n      }\n    }\n#endif\n")
    a = rep(a, old, new)
    s0 = "      int qv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n      float ps[8];"
    s1 = "      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n      CTA_SYNC();  // red is reused by the next token\n"
    assert a.count(s0) == 1 and a.count(s1) == 1
    i = a.index(s0); j = a.index(s1) + len(s1)
    new = r'''#if OKC_FIN1 & 2
#if !OKC_YBF
#error "OKC_FIN1 needs OKC_YBF"
#endif
      {
        const float* ssb = p.ybuf + (size_t)EMAX * HD / 2;
        const int qs = S.pos[tk * 8 + (tid >> 6)];
        float4 sv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qs >= 0) sv = __ldcg(reinterpret_cast<const float4*>(ssb + (size_t)qs * 256) + (tid & 63));
        int qv[8];
        float wv[8];
        uint4 yr[8];
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          qv[sl] = S.pos[tk * 8 + sl];
          wv[sl] = 0.f;
          yr[sl] = make_uint4(0u, 0u, 0u, 0u);
          if (qv[sl] >= 0) {
            wv[sl] = __ldg(p.topk_w + tk * 8 + sl);
            yr[sl] = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          }
        }
        float s = sv.x + sv.y + sv.z + sv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[warp] = s;
        CTA_SYNC();
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[sl] >= 0) {
            const float r = wv[sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
            const uint32_t* yw = reinterpret_cast<const uint32_t*>(&yr[sl]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yw[i]));
              acc[2 * i] += r * f.x; acc[2 * i + 1] += r * f.y;
            }
          }
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
          ow[i] = *reinterpret_cast<uint32_t*>(&v);
        }
        *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
        CTA_SYNC();  // red is reused by the next token
      }
#else
'''
    a = a[:i] + new + a[i:j] + "#endif\n" + a[j:]
    return a

# ---- moe_ovl_v1 round 5: phase-A cost table flattened toward the measured per-ntt stage times (fat CTAs end A
# 20-30 us before skinny ones, deterministic; r4)
S = load_sub("/w/run/sub.py", "sub")
a = patch_common(S.CUDA_SRC)
CA = [112, 135, 140, 158, 173, 185, 200, 207]
def fa(f): return [round(112 + (c - 112) * f) for c in CA]
ARMS_T = {"ctl": 1.0, "a08": 0.8, "a07": 0.7, "a06": 0.6, "a05": 0.5}
P = None
w = R.generate_weights(R.WEIGHTS_SEED)
B = Bench(S, w)
ARMS = {"base": B.MAIN}
for n, f in ARMS_T.items():
    S.ALT_SRCS[n] = set_tables(a, A=fa(f)); ARMS[n] = B.MAIN + ("@" + n,)
    print("TABLE", n, fa(f), flush=True)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
todo = list(ARMS.values())[1:]
futs = [pool.submit(S._ext, d) for d in todo]
st = S.prepare(w); print("PREPARED", el(), flush=True)
for f, d in zip(futs, todo): f.result()
B.setup(st)
for bs, dist in ((33, "normal"), (128, "skewed"), (257, "normal"), (64, "normal")):
    d = R.generate_input(bs, "ep8", 777 + bs, dist)
    for n, df in ARMS.items(): print("CHECK", bs, dist, n, B.check(df, d), flush=True)
SEEDS = {32: [321, 323, 325], 128: [5041754901133868307, 1281, 1282, 1284, 1286], 256: [5041887071557145533, 2561, 2563, 2566, 2568]}
rows = []
for bs in (128, 256, 32):
    for seed in SEEDS[bs]:
        if el() > 820: print("SKIP_BUDGET", flush=True); break
        d = R.generate_input(bs, "ep8", seed); rs = route_stats(d)
        row = dict(bs=bs, seed=seed, D=rs["D"], **B.paired(ARMS, d, reps=14))
        rows.append(row); print("ROW", json.dumps(row), flush=True)
for bs in (32, 128, 256):
    rr = [r for r in rows if r["bs"] == bs]
    if rr: print("SUMMARY", bs, {n: round(math.exp(statistics.mean(math.log(r[n]) for r in rr)), 4) for n in rr[0] if n.startswith("r_")},
                 "base_us", round(statistics.mean(r["base"] for r in rr), 1), flush=True)
print("DONE", el(), flush=True)
