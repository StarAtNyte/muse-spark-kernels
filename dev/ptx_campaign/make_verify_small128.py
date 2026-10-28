"""Add the validated local small-batch tile128 path to the promoted parent."""
from pathlib import Path
import ast
parent=Path('experiments/attention_verify_onehead_s2/submission.py').read_text()
small=Path('dev/ptx_campaign/attention_verify_tile128.py').read_text()
a=small.index('_WG_CUDA =');b=small.index('_WG_CPP =',a)
cuda=small[a:b].replace('_WG_CUDA =','_WG_SMALL_CUDA =',1)
a=small.index('def _wg_module():');b=small.index('_FD_CUDA =',a)
functions=small[a:b]
functions=functions.replace('_wg_module','_wg_small_module').replace('_WG_MOD','_WG_SMALL_MOD').replace('_WG_CUDA','_WG_SMALL_CUDA').replace('_wg_forward','_wg_small_forward')
old="    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window']"
new="""    if (cfg['q_len'] == 8 and not cfg['window'] and cfg['batch'] <= 4
            and cfg['num_kv_heads'] == 2 and cfg['num_q_heads'] == 16
            and 32768 <= ctx_bound <= (1 << 20) + 64):
        qkv, q = _fd_prepare(data)
        splits = max(32, 128 // triton.next_power_of_2(cfg['batch']))
        return _wg_small_forward(data, q, qkv, splits)
    if (_WG_ON and cfg['q_len'] == 8 and not cfg['window']"""
assert parent.count(old)==1
parent=parent.replace(old,new)
parent+='\n\n# Small full-context verification: independent-head 128-key tiles.\n'+cuda+'\n_WG_SMALL_MOD = None\n\n'+functions
ast.parse(parent)
Path('dev/ptx_campaign/attention_verify_small128.py').write_text(parent)
