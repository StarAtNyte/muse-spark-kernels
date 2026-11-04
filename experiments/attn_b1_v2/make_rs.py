"""Row-split replicas for one-request q_len 8 sliding: each tile's CTA is launched RS times (grid z = pairs * RS);
the replicas do the same tile work but only replica r's warps (warp / (8 / RS) == r) add their rows to the workspace,
so each SM issues 1/RS of the red.global.add traffic that dominates the q_len 8 merge (0.86 us measured)."""
import pathlib
from make_cand import rep
H = pathlib.Path(__file__).resolve().parent
def rs(s, default_rs=2):
    s = rep(s, '  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n',
            '  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n'
            '  int rs;               // row-split replicas per (request, kv pair, split): grid z = pairs * rs (1: none)\n')
    s = rep(s, '  const int pair = blockIdx.z;\n  if (!FLAT) {', '  const int pair = (int)blockIdx.z / p.rs;\n  if (!FLAT) {')
    s = rep(s, '  const int bp = b * pairs + pair;\n  const int ntiles = ONE',
            '  const int bp = b * pairs + pair;\n'
            '  const int rsel = FLAT ? 0 : (int)blockIdx.z - pair * p.rs;   // row-split replica of this CTA\n'
            '  const int ntiles = ONE')
    s = rep(s, '        if (pos >= start && pos < end) {\n          const int64_t row = S.cloc[jj];',
            '        if (pos >= start && pos < end && rsel == 0) {   // (one replica writes the new cache rows)\n          const int64_t row = S.cloc[jj];')
    s = rep(s, '  if (M_CL) {\n    cp_wait<0>();',
            '  // row-split replicas: only this replica\'s share of the warps (their rows) add to the workspace\n'
            '  if (L == 8 && p.rs > 1 && warp / (8 / p.rs) != rsel) do_red = false;\n'
            '  if (M_CL) {\n    cp_wait<0>();')
    s = rep(s, '      if (old + mine == n_keys) {', '      if (old + mine == n_keys * p.rs) {   // (every replica counts its keys)')
    s = rep(s, 'if (v >= n_keys) break;', 'if (v >= n_keys * p.rs) break;', 2)
    s = rep(s, '    S.flag = (old + mine == n_keys);', '    S.flag = (old + mine == n_keys * p.rs);')
    s = rep(s, '    const int split_idx = (int)blockIdx.x;\n    int nact = fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys',
            '    const int split_idx = (int)blockIdx.x * p.rs + rsel;\n'
            '    int nact = fdiv_i(n_keys + chunk - 1, chunk) * p.rs;    // CTAs with keys (splits x replicas)')
    s = rep(s, '  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));',
            '  p.rs = (flat || L != 8) ? 1 : std::max(1, (int)((tma >> 20) & 15));\n'
            '  TORCH_CHECK(p.rs == 1 || p.rs == 2 || p.rs == 4 || p.rs == 8, "rs must be 1, 2, 4 or 8");\n'
            '  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2 * p.rs));')
    s = rep(s, '    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),\n',
            '    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),\n'
            f'    (1, 8, True, 1): dict(rs={default_rs}, coop=1),   # one request, q_len 8 sliding: row-split replicas (attn_b1_v2)\n')
    s = rep(s, '    tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT)',
            '    rs = int(plan.get("rs", 1)) if (L == 8 and not flat and cs <= 1) else 1\n'
            '    tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT) | (rs << RS_SHIFT)')
    s = rep(s, 'PDL_BIT = 16   # bit 16: programmatic dependent launch\n',
            'PDL_BIT = 16   # bit 16: programmatic dependent launch\nRS_SHIFT = 20  # bits 20-23: row-split replicas (q_len 8, per-request grids)\n')
    return s.replace('name=f"okc_attn_v298', 'name=f"okc_attn_b1rs')
if __name__ == '__main__':
    (H / 'cand_rs.py').write_text(rs((H / 'parent.py').read_text())); print('ok')
