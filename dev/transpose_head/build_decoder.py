"""Reuse the selected standalone attention kernel in the whole decoder."""
import ast
from pathlib import Path
attention=Path('dev/ptx_campaign/attention_transpose_selected.py').read_text()
s=Path('experiments/decoder_wm_v3/submission.py').read_text()
assert '_SP_CPP' not in s
s+='\n\n'+attention[attention.index('_SP_CPP ='):]
old="    fused_slide = _SLIDE_STAGES and cfg['window'] and length == 1 and cfg['page_size'] & (cfg['page_size'] - 1) == 0"
new="""    use_transposed = (length == 1 and 8 <= batch <= 64 and hq == 16 and hkv == 2 and d == 64
                      and data['page_table'].shape[1] * cfg['page_size'] <= (1 << 20) + 64
                      and cfg['page_size'] & (cfg['page_size'] - 1) == 0)
    fused_slide = (not use_transposed and _SLIDE_STAGES and cfg['window'] and length == 1
                   and cfg['page_size'] & (cfg['page_size'] - 1) == 0)"""
assert s.count(old)==1;s=s.replace(old,new)
old="    if fused_slide:\n        o = _slide_forward(data, _SLIDE_STAGES, qkv).view(tokens, hq*d)"
new="    if use_transposed:\n        o = _sp_forward(data, q, qkv).view(tokens, hq*d)\n    elif fused_slide:\n        o = _slide_forward(data, _SLIDE_STAGES, qkv).view(tokens, hq*d)"
assert s.count(old)==1;s=s.replace(old,new)
ast.parse(s);Path('dev/ptx_campaign/decoder_transpose_selected.py').write_text(s)
