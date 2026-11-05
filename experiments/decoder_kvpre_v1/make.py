"""Build decoder_kvpre_v1/submission.py from decoder_readable_fp32_ybuf_v4: per-request attention kernels launched
under PDL issue the page indices AND the first STAGES-1 tiles of cached K/V before griddepcontrol.wait; q, the new
k/v and the gate rows (the QKV GEMM's outputs) are read with direct loads after it. Runtime switch: tma bit 17
(Python OKC_PDLKV_RT), so a same-binary control is possible."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, (c, old[:80])
    src = src.replace(old, new)

rep("  int pdlf;             // launched with programmatic dependent launch (OKC_PDLKV prologue order)\n",
    "  int pdlf;             // launched with programmatic dependent launch (OKC_PDLKV prologue order)\n"
    "  int pdlkv;            // (kvpre) per-request kernels: cached K/V of the first tiles issued before griddepcontrol.wait\n")
rep("  const bool pdl = ((tma >> 16) & 1) && major >= 9;   // programmatic dependent launch (off by default)\n",
    "  const bool pdl = ((tma >> 16) & 1) && major >= 9;   // programmatic dependent launch (off by default)\n"
    "  p.pdlkv = (pdl && ((tma >> 17) & 1)) ? 1 : 0;\n")

old_pro = src[src.index("  {\n  if constexpr (!WS) {\n#pragma unroll\n    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);"):
              src.index("  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----")]
assert src.count(old_pro) == 1
consts = """  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tidL == 10)   // the coop barrier's flag / epoch, early
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
"""
assert consts in old_pro, 'consts'
gatepf = old_pro[old_pro.index("  if (split == 0 && (FLAT || !M_CL)) {  // warm L2"):]
head = old_pro[:old_pro.index("  pdl_wait();   // qkv (q, new k / v, gate)")]
body = old_pro[old_pro.index("  pdl_wait();   // qkv (q, new k / v, gate)"):old_pro.index(gatepf)]
assert head.endswith("\n") and consts in body
head = head.replace("  cp_commit();\n", "  if (pkv) {   // (kvpre) the constants join the page-index group: nothing in it is the QKV GEMM's output\n"
                    + "\n".join("  " + l if l else l for l in consts.split("\n")) + "}\n  cp_commit();\n", 1)
head = "  // (kvpre) PDL-launched per-request kernels: the cached K/V rows never depend on the preceding QKV GEMM\n" \
       "  const bool pkv = !WS && p.pdlkv != 0;\n" + head
new_pro = head + "  if (!pkv) {\n" + body + gatepf.replace("  if (split == 0", "  if (split == 0", 1) + "  }\n"
src = src.replace(old_pro, new_pro)

rep("  cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  if constexpr (!FLAT && (MODE & 8) != 0) {",
    "  if (pkv) cp_wait<0>(); else cp_wait<IDXF ? 1 : 0>();\n  csync<WS, NT>();\n  if constexpr (!FLAT && (MODE & 8) != 0) {")

direct = """  if (pkv) {
    // (kvpre) the first tiles' cached K/V are in flight: now wait for the QKV GEMM, then q, the new k / v rows and the
    // cluster gate slice by direct (L2) loads, all issued before any is stored (no cp.async group: the main loop's
    // group count stays the STAGES-1 K/V groups)
    pdl_wait();
    pdl_launch_dependents();
    constexpr int QN = (L * 16 * 8 + NT - 1) / NT, KN = (L * 4 * 8 + NT - 1) / NT;
    uint4 qr[QN], kr[KN];
#pragma unroll
    for (int i = 0; i < QN; ++i) {
      const int e = tidL + i * NT;
      if ((L * 16 * 8) % NT == 0 || e < L * 16 * 8) {
        const int rr = e >> 3, ch = e & 7, j = rr >> 4, hh = rr & 15;
        qr[i] = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8));
      }
    }
    if (has_new) {
#pragma unroll
      for (int i = 0; i < KN; ++i) {
        const int e = tidL + i * NT;
        if ((L * 4 * 8) % NT == 0 || e < L * 4 * 8) {
          const int rr = e >> 3, ch = e & 7;
          const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
          const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
          kr[i] = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col));
        }
      }
    }
#pragma unroll
    for (int i = 0; i < QN; ++i) {
      const int e = tidL + i * NT;
      if ((L * 16 * 8) % NT == 0 || e < L * 16 * 8)
        *reinterpret_cast<uint4*>(&S.qst()[(e >> 3) * 64 + (e & 7) * 8]) = qr[i];
    }
    if (has_new) {
#pragma unroll
      for (int i = 0; i < KN; ++i) {
        const int e = tidL + i * NT;
        if ((L * 4 * 8) % NT == 0 || e < L * 4 * 8)
          *reinterpret_cast<uint4*>(&S.kvst[(e >> 3) * 64 + (e & 7) * 8]) = kr[i];
      }
    }
    if (M_CL) {
      const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
      for (int e = threadIdx.x; e < RPC * 8; e += NT) {
        const int rl = e >> 3, ch = e & 7;
        const int row = row0 + rl, j = row >> 4, hh = row & 15;
        *reinterpret_cast<uint4*>(&S.gst[rl * 64 + ch * 8]) = __ldcg(reinterpret_cast<const uint4*>(
            p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8));
      }
    }
""" + "\n".join("  " + l if l else l for l in gatepf.rstrip("\n").split("\n")) + "\n  }\n"
rep("  DBG(8);\n  }\n\n  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n  }\n",
    "  DBG(8);\n  }\n\n" + direct +
    "  if (IDXF && !pkv) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n  }\n")

rep('_ATT7 = load_inline(name=f"muse_att_v298w1pf_{arch}"', '_ATT7 = load_inline(name=f"muse_att_v298w1pfkv_{arch}"')
rep("OKC_PDL = 0      # launch with programmatic dependent launch",
    "OKC_PDLKV_RT = 1  # (kvpre) PDL per-request attention: first K/V tiles before griddepcontrol.wait (tma bit 17)\nOKC_PDL = 0      # launch with programmatic dependent launch")
rep("(int(use_pdl) << 16)\n", "(int(use_pdl) << 16) | (int(bool(OKC_PDLKV_RT)) << 17)\n")
open(sys.argv[2], 'w').write(src)
print('ok', len(src))
