from pathlib import Path
import ast
s=Path('experiments/attention_verify_onehead_s2/submission.py').read_text()
a=s.index('def _wg_forward('); b=s.index('_FD_CUDA =',a)
x=s[a:b]
x=x.replace("span = data['page_table'].shape[1] * cfg['page_size']", "span = cfg['window'] + length - 1 if cfg['window'] else data['page_table'].shape[1] * cfg['page_size']")
x=x.replace("length, hq, cfg['page_size'], 0, splits, chunk,", "length, hq, cfg['page_size'], cfg['window'], splits, chunk,")
s=s[:a]+x+s[b:]
s += '\n_WG_SLIDE_SPLITS = 4\n'
old="    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window']"
new="""    if (cfg['q_len'] == 8 and cfg['window'] and cfg['num_kv_heads'] == 2
            and cfg['num_q_heads'] == 16):
        qkv, q = _fd_prepare(data)
        return _wg_forward(data, q, qkv, _WG_SLIDE_SPLITS)
    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window']"""
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s)
Path('dev/ptx_campaign/attention_verify_sliding.py').write_text(s)
