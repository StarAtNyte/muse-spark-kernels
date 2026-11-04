"""Build decoder_stack_v1/submission.py from decoder_bf16_v1/submission.py (rs128 BF16 split reduction at T=128/256,
BF16 ybuf + fallbacks, g16b attention) plus L2PF: the sk2 qkv kernel, after issuing its last weight stage, prefetches
the o projection's 12-bit weight stream into L2 (cp.async.bulk.prefetch.L2; weights do not depend on activations), so
the o kernel after the latency-bound attention reads L2 instead of HBM. Variants for A/B: python make_submission.py [tag cfg]."""
import pathlib, sys, hashlib
E = pathlib.Path(__file__).resolve().parent.parent
s = (E / "decoder_bf16_v1/submission.py").read_text()

def once(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a))
    return s.replace(a, b)

# kernel: the qkv kernel's producer lane issues the prefetch of region (dbg, dg << 12) split evenly over the grid.
# dg / dbg are the diagnostics arguments, unused by production builds (SK2_DIAG 0); gle builds pass 0 / nullptr.
s = once(s, '      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued\n',
'''      if (!SK2_DIAG && dg > 0 && dbg != nullptr) {
        // L2PF: the next projection's weight stream into L2 while this kernel drains and attention runs (no data
        // dependency: weights only). Region = dg * 4 KB from dbg, split evenly over the grid, 64 KB per request.
        const size_t tot = (size_t)dg << 12;
        const size_t per = ((tot + gridDim.x - 1) / gridDim.x + 4095) & ~(size_t)4095;
        const size_t a = (size_t)blockIdx.x * per, e = a + per < tot ? a + per : tot;
        for (size_t o = a; o < e; o += 65536) {
          const uint32_t sz = (uint32_t)(e - o < 65536 ? e - o : 65536);
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(reinterpret_cast<const uint8_t*>(dbg) + o), "r"(sz) : "memory");
        }
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued
''')
# host: base and narrow qkv launches pick up the current prefetch region
s = once(s, '''    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)''',
'''    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    if layer == "qkv_gate" and _L2PF_CUR is not None and dbg is None:
        dg, dbg = _L2PF_CUR
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)''')
s = once(s, """ ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)""",
""" ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 if _L2PF_CUR is not None and dbg is None:dg,dbg=_L2PF_CUR
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)""")
s = once(s, "bfp_SK2_PDL = 0   # decode_layer:", "_L2PF_CUR = None   # (4 KB units, weight stream) the next qkv sk2 launch prefetches into L2\nbfp_SK2_PDL = 0   # decode_layer:")
CFG = sys.argv[2] if len(sys.argv) > 2 else '{"s": (1, 8, 16, 32)}'
s += f'''

# ---- L2PF: per (layer kind, T), the qkv kernel prefetches the o projection's weight stream into L2 ("s" sliding, "f" full)
L2PF_CFG = {CFG}
L2PF_HITS = {{}}
_l2pf_parent_kernel_fn = kernel_fn


def kernel_fn(data, state):
    global _L2PF_CUR
    cfg = data["config"]
    T = cfg["tokens"]
    _L2PF_CUR = None
    bst = state.get("bfp") if isinstance(state, dict) else None
    if bst is not None and T in L2PF_CFG.get("s" if cfg["window"] > 0 else "f", ()):
        ent = bst.get("sk2", {{}}).get(bfp__sk2_name("o", T))
        if ent is not None:
            w = ent[1]
            _L2PF_CUR = ((w.numel() * w.element_size()) >> 12, w)
            L2PF_HITS[T] = L2PF_HITS.get(T, 0) + 1
    try:
        return _l2pf_parent_kernel_fn(data, state)
    finally:
        _L2PF_CUR = None
'''
POOL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
s += f'''

# ---- L2PF pooler prefix: at T <= 8 the o stream and the pooler stream live in one buffer, so the qkv kernel's prefetch
# region can run past the o weights into the first L2PF_POOL bytes of the pooler (stage-major: the first stages of every CTA)
L2PF_POOL = {POOL}
_l2pf_parent_prepare = prepare


def prepare(weights):
    st = _l2pf_parent_prepare(weights)
    bst = st.get("bfp") if isinstance(st, dict) else None
    try:
        if L2PF_POOL > 0 and bst is not None:
            sk = bst["sk2"]
            on = bfp__sk2_name("o", 8)
            eo, ep = sk[on], sk["pooler"]
            ob = eo[1].numel() * eo[1].element_size()
            pb = ep[1].numel() * ep[1].element_size()
            obp = (ob + 4095) // 4096 * 4096
            buf = torch.empty(obp + pb, dtype=torch.uint8, device=eo[1].device)
            buf[:ob].copy_(eo[1].contiguous().view(-1).view(torch.uint8))
            buf[obp:].copy_(ep[1].contiguous().view(-1).view(torch.uint8))
            nw_o = buf[:ob].view(eo[1].dtype).view(eo[1].shape)
            nw_p = buf[obp:].view(ep[1].dtype).view(ep[1].shape)
            sk[on] = (eo[0], nw_o) + tuple(eo[2:])
            sk["pooler"] = (ep[0], nw_p) + tuple(ep[2:])
            st["l2pf_buf"] = (buf, (obp + min(L2PF_POOL, pb)) >> 12)
            torch.cuda.synchronize()
            print(f"[l2pf] o {{ob}} B + pooler prefix {{min(L2PF_POOL, pb)}} of {{pb}} B in one buffer", flush=True)
    except Exception as ex:
        print(f"[l2pf] relocation failed: {{ex!r}}"[:200], flush=True)
    return st
'''
s = s.replace('''            w = ent[1]
            _L2PF_CUR = ((w.numel() * w.element_size()) >> 12, w)''', '''            w = ent[1]
            _L2PF_CUR = ((w.numel() * w.element_size()) >> 12, w)
            if T <= 8 and state.get("l2pf_buf") is not None:
                _L2PF_CUR = (state["l2pf_buf"][1], state["l2pf_buf"][0])''')
tag = sys.argv[1] if len(sys.argv) > 1 else ""
dst = E / "decoder_stack_v1" / (f"var{tag}.py" if tag else "submission.py")
dst.write_text(s)
import ast; ast.parse(s)
print(dst.name, len(s.encode()), "bytes sha", hashlib.sha256(s.encode()).hexdigest()[:16])
