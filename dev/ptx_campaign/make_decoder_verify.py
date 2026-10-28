"""Integrate the validated standalone WGMMA verification path into decoder."""
import ast
from pathlib import Path
attn=Path('experiments/attention_verify_onehead_s2/submission.py').read_text()
s=Path('experiments/decoder_wm_v3/submission.py').read_text()
start=attn.index('_WG_CUDA =')
end=attn.index('_FD_CUDA =', start)
wg=attn[start:end].replace('"muse_wg5_onehead_s2"','"muse_decoder_wg5_onehead_s2"')
old='''    elif fd is not None and cfg['page_size'] & (cfg['page_size'] - 1) == 0:
'''
new='''    elif (length == 8 and not cfg['window'] and batch >= 8 and hkv == 2 and hq == 16
          and data['page_table'].shape[1] * cfg['page_size'] <= (1 << 20) + 64):
        o = _wg_forward(data, q, qkv, 16 if batch <= 24 else 32).view(tokens, hq*d)
    elif fd is not None and cfg['page_size'] & (cfg['page_size'] - 1) == 0:
'''
assert s.count(old)==1
s=s.replace(old,new)
s += '\n\n# Validated independent-KV-head WGMMA verification path.\n' + wg
ast.parse(s)
Path('dev/ptx_campaign/decoder_verify_onehead_s2.py').write_text(s)
