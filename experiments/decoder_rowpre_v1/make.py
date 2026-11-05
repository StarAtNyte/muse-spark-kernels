"""Build decoder_rowpre_v1/submission.py from decoder_readable_fp32_ybuf_v4: the T = 17..32 (and T > 32) row
boundary gets runtime knobs (warps / vector width, stream preload before griddepcontrol.wait for the first boundary,
whose stream is the layer input) so one binary can be screened against a same-code control."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, (c, old[:80])
    src = src.replace(old, new)

rep('''def _graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    # our _boundary_preload_kernel (r20, 1 < m <= 32): weights before the wait, router copy removed
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c)
''', '''def _graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr,
                            SPRE: _gl.constexpr = False):
    # our _boundary_preload_kernel (r20, 1 < m <= 32): weights before the wait, router copy removed
    # (rowpre) SPRE: the residual stream (a layer input, never a graph-internal product) is loaded before the wait too
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    if SPRE:
        s = _gl.load(Stream + row * H + c)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    if not SPRE:
        s = _gl.load(Stream + row * H + c)
''')

rep('''    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                      g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,
                                      num_warps=32, launch_pdl=True)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
''', '''    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        v, nw = ROWPRE_VNW
        for _ in range(ROWPRE_DUP):
            _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                          g["beta" + pfx], h_out, s_out, HIDDEN, width, post, v, nw,
                                          ROWPRE_SPRE and not post, num_warps=nw, launch_pdl=True)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        for _ in range(ROWPRE_DUP):
            if ROWPRE_BIG_PRE:
                _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                              g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4,
                                              16 if T > 192 else 32, ROWPRE_SPRE and not post,
                                              num_warps=16 if T > 192 else 32, launch_pdl=True)
            else:
                _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                          g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                          num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
''')
rep("def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):\n",
    "ROWPRE_VNW = (4, 32)    # (rowpre) T = 17..32 row boundary: (vector width, warps); parent (4, 32)\n"
    "ROWPRE_SPRE = False     # (rowpre) first boundary: residual stream loaded before griddepcontrol.wait\n"
    "ROWPRE_BIG_PRE = False  # (rowpre) T > 32: the weight-preloading variant instead of _graft_row_boundary\n"
    "ROWPRE_DUP = 1          # (rowpre) diagnostic only: launch the boundary this many times (idempotent)\n\n\n"
    "def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):\n")
for name in ("b1", "b2c"):
    rep(f'  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft {name}: rows must be 1..16");\n',
        f'  if(a.rows<1 || a.rows>256) throw std::runtime_error("graft {name}: rows must be 1..256");\n')
rep('''    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        v, nw = ROWPRE_VNW
''', '''    if (ROWPRE_CLUS if T <= 32 else ROWPRE_CLUSBIG) and br.dtype == torch.bfloat16 and width == HIDDEN and nu == HIDDEN:
        # (rowpre) the 8-CTA cluster boundary (same lane mapping and FP order as the preload Gluon kernel) at
        # T = 17..256: 8 x T CTAs of 128 threads spread the row traffic over every SM instead of T CTAs
        _graft_al(br, stream, s_out, h_out)
        for _ in range(ROWPRE_DUP):
            if post:
                _graft_glue_module().b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            else:
                _graft_glue_module().b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out, s_out, False)
        return True
    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        v, nw = ROWPRE_VNW
''')
rep("ROWPRE_DUP = 1 ", "ROWPRE_CLUS = False     # (rowpre) T = 17..32: the 8-CTA cluster boundary (b1 / b2c)\n"
    "ROWPRE_CLUSBIG = False  # (rowpre) T > 32: the 8-CTA cluster boundary\nROWPRE_DUP = 1 ")
open(sys.argv[2], 'w').write(src)
print('ok', len(src))
