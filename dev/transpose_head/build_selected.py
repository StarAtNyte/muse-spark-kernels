"""Integrate the independently validated dense-transposed decode layout."""
import ast,runpy,subprocess
from pathlib import Path
x=runpy.run_path('dev/transpose_head/decode.py')
cuda=x['CUDA'].replace('trans_decode<2>','trans_decode<3>').replace('2*8192','3*8192')
Path('dev/transpose_head/decode_selected.py').write_text('CPP='+repr(x['CPP'])+'\nCUDA='+repr(cuda)+'\n')
out='dev/ptx_campaign/attention_transpose_selected.py'
subprocess.run(['python','dev/sparse_head/build_candidate.py','--kernel','dev/transpose_head/decode_selected.py','--output',out,'--module','muse_dense_transposed_decode_v1'],check=True)
s=Path(out).read_text().replace("cfg['q_len'] == 1 and cfg['batch'] <= 64","cfg['q_len'] == 1 and 8 <= cfg['batch'] <= 64")
s=s.replace('_SP_SPLITS_OVERRIDE = None\n','')
s=s.replace("    splits = _SP_SPLITS_OVERRIDE\n    if splits is None:\n        splits = (64 if batch == 1 else 8 if batch <= 24 else 4) if window else (256 if batch == 1 else 64 if batch <= 24 else 32)\n", "    # Target enough CTAs to fill H200 while keeping each chunk economical.\n    splits = max(4 if window else 8, triton.next_power_of_2(triton.cdiv(256 if window else 512, batch)))\n")
s=s.replace("    chunk = triton.cdiv(triton.cdiv(span, splits), 32) * 32", "    splits = max(splits, triton.cdiv(span, 8192))\n    chunk = triton.cdiv(triton.cdiv(span, splits), 32) * 32")
ast.parse(s);Path(out).write_text(s)
