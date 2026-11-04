"""Build the candidate from the exact scored FP32 source by four text edits inside CUDA_SRC plus one PS_SHAPES entry.
Usage: python3 make_candidate.py PARENT.py OUT.py"""
import sys, hashlib
parent, out = sys.argv[1], sys.argv[2]
s = open(parent).read()
cut = s.index("VAR_SRC = r'''")  # CUDA_SRC lies before this; VAR_SRC (tiny builds) is left alone
head, tail = s[:cut], s[cut:]
def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:70])
    return text.replace(old, new)
head = sub(head, "constexpr int WGS_NS = 3;",
    "#ifndef OKC_WGSNS\n#define OKC_WGSNS 3   // weight ring slots; the bs<=32 build passes 4\n#endif\nconstexpr int WGS_NS = OKC_WGSNS;")
head = sub(head, "constexpr int WGS_XS = WGS_NT * 4096;",
    "#ifndef OKC_WGSXR\n#define OKC_WGSXR OKC_FT_T   // rows an x slot holds; a build may pass fewer only if no job has more tokens\n#endif\n"
    "static_assert(OKC_WGSXR % 8 == 0 && OKC_WGSXR <= OKC_FT_T, \"x slot rows\");\n"
    "constexpr int WGS_XS = (OKC_WGSXR / 8) * 4096;")
head = sub(head, "constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;",
    "constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = 2 * WGS_NS + WGS_NX;")
head = sub(head, "if (n >= 2 && p.M <= 64) { const int pn = n - 2;",
    "if (n >= WGS_NS - 1 && p.M <= 64) { const int pn = n - (WGS_NS - 1);")
tail = sub(tail, "PS_SHAPES = {8: ('tiny2', {}, 't2_wsi')}",
    "# bs 32: every expert has <= 32 tokens (a token routes to an expert at most once), so an x slot needs 32 rows (16 KB);\n"
    "# the 48 KB saved pay for a 4th weight stage (4 x 36 KB + 3 x 16 KB = 192 KB of rings, 3 weight stages in flight).\n"
    "PS_SHAPES = {8: ('tiny2', {}, 't2_wsi'), 32: ('main', {'OKC_WGSNS': 4, 'OKC_WGSXR': 32}, None)}")
open(out, "w").write(head + tail)
print(hashlib.sha256(open(parent,'rb').read()).hexdigest(), hashlib.sha256(open(out,'rb').read()).hexdigest())
