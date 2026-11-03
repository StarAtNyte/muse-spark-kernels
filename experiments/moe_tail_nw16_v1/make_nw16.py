"""Build submission.py from the baseline (experiments/moe_keell0renz_compare_v1/submission.py): 16 compute warps in the
TINY (M <= 4) and TINY2 / bs8 builds. The warp-specialized skinny ring gets the NW = 16 scale addressing the static ring
already has; NW = 8 builds (MID) compile the same text as before."""
src = open('../moe_keell0renz_compare_v1/submission.py').read()
a = src.index("VAR_SRC = r'''"); b = src.index("BW_SRC = r'''")
head, var, tail = src[:a], src[a:b], src[b:]
i0 = var.index("__device__ void skinny_ring_ws(")
i1 = var.index("__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {")
fn = var[i0:i1]
old = "    const uint4 sc = sc_load(slot, warp, lane);\n"
assert fn.count(old) == 1
fn = fn.replace(old, """#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    static_assert(!OKC_SC6S, "NW = 16 skinny units read the plain scale layout");
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = sc_load(slot, warp, lane);
#endif
""")
var = var[:i0] + fn + var[i1:]
for name in ("TINY_DEFINES = (", "TINY2_DEFINES = ("):
    old = name + '"-DOKC_NW=8", '
    assert head.count(old) == 1
    head = head.replace(old, name + '"-DOKC_NW=16", ')
head = head.replace("# okc: problem=muse_moe_w4a16\n",
    "# okc: problem=muse_moe_w4a16\n# tail_nw16: the TINY (M <= 4) and TINY2 / bs8 builds run 16 compute warps (was 8): their skinny units were\n"
    "#   compute-latency bound (~1 us per resident 36 KB unit at 2 warps per scheduler); the WS ring reads the NW = 16 scale layout.\n", 1)
open('submission.py', 'w').write(head + var + tail)
print('ok')
