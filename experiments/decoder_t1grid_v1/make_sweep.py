"""Build sweep.py = decoder_stack_v1 + extra tokr-8 sk2 configs (appended; earlier indices unchanged), selectable at run
time through bfp_SK2_PICK. The qkv narrow (T=2..8) launcher is bypassed for any non-default qkv pick."""
import sys
src = open('../decoder_stack_v1/submission.py').read()
EXTRA = [  # (name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL)
    ('qkv8n2_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),
    ('qkv8w8_r176', 'qkv_gate', 2304, 24, 3, 1, 8, 1, 1, 1),
    ('qkv8c2_r100', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 2),
    ('qkv8r16_r100', 'qkv_gate', 2304, 16, 2, 1, 4, 2, 1, 1),
    ('qkv8r16w8_r100', 'qkv_gate', 2304, 16, 2, 1, 8, 1, 1, 1),
    ('qkv8r16b_r100', 'qkv_gate', 2304, 16, 1, 2, 4, 2, 1, 1),
    ('qkv8r8_r72', 'qkv_gate', 2304, 8, 1, 1, 8, 1, 1, 1),
    ('qkv8r32c2_r100', 'qkv_gate', 2304, 32, 4, 1, 4, 2, 1, 2),
    ('slice8c2_r100', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 2),
    ('slice8u1_r192', 'unpooler', 1024, 8, 1, 1, 8, 1, 2, 1),
    ('slice8w4_r192', 'unpooler', 1024, 8, 1, 1, 4, 2, 2, 1),
    ('slice8n1_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 1, 1),
    ('slice8w16_r192', 'unpooler', 1024, 8, 1, 1, 16, 1, 2, 1),
    ('pool8a_r100', 'pooler', 4096, 16, 1, 2, 4, 2, 1, 1),
    ('pool8b_r72', 'pooler', 4096, 16, 2, 1, 4, 2, 1, 1),
    ('pool8r8_r72', 'pooler', 4096, 8, 1, 1, 8, 1, 1, 1),
    ('pool8n2_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 2, 1),
    ('pool8c2_r100', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 2),
]
anchor = "    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "    # decoder_t1grid_v1 sweep (tokr 8 only)\n" + "".join(f"    {c!r},\n" for c in EXTRA))
alt = "bfp_SK2_ALT = {" + ", ".join(f"({c[0]!r}, 8)" for c in EXTRA) + ", ('pooler', 8)}\n"
a2 = "bfp_SK2_PICK = {  # (layer, tokr) -> config name\n"
assert src.count(a2) == 1
src = src.replace(a2, alt + a2)
a3 = "            if bfp_SK2_PICK.get((layer, tokr)) != name:\n"
assert src.count(a3) == 1
src = src.replace(a3, "            if bfp_SK2_PICK.get((layer, tokr)) != name and (name, tokr) not in bfp_SK2_ALT:\n")
a4 = " if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n"
assert src.count(a4) == 1
src = src.replace(a4, " if layer!='qkv_gate' or not 2<=m<=8 or bfp__sk2_name(layer,m)!='qkv8_r176':return _DE_RUN(x,layer,m,n,k,state,dg,dbg)\n")
open('sweep.py', 'w').write(src)
print('ok', len(src))
