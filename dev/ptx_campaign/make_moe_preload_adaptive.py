"""Keep four-chunk prefetch for NCH=1; reduce it only for NCH>=2."""
from pathlib import Path
import ast
base=Path('experiments/moe_wm_v22/submission.py').read_text()
s=Path('dev/ptx_campaign/moe_preload2.py').read_text()
marker='  if (RING == 0) {'
a=s.index(marker);b=s.index('  } else if (RING == 2)',a)
pa=base.index(marker);pb=base.index('  } else if (RING == 2)',pa)
original=base[pa+len(marker):pb]
reduced=s[a+len(marker):b]
block=marker+'\n    if constexpr (NCH >= 2) {\n'+reduced+'    } else {\n'+original+'    }\n'
s=s[:a]+block+s[b:]
ast.parse(s)
Path('dev/ptx_campaign/moe_preload_adaptive.py').write_text(s)
p=Path('dev/modal/drv_moe_preload2.py').read_text().replace("('preload2', candidate_cuda, [])", "('preload_adaptive', candidate_cuda, [])")
p=p.replace('for batch in (1, 8, 16, 32, 128):','for batch in (8, 16, 32):').replace('for seed in (419, 2387):','for seed in (101, 419, 2387, 7919, 15241, 49157):')
old="            data = R.generate_input(batch, 'ep8', seed + batch)\n"
new=old+'''            ids = data['topk_ids']
            local = ids[(ids >= 0) & (ids < 32)].long()
            counts = torch.bincount(local, minlength=32)
            print('ROUTING', batch, seed, counts.cpu().tolist(), flush=True)
'''
assert p.count(old)==1
p=p.replace(old,new)
ast.parse(p)
Path('dev/modal/drv_moe_preload_adaptive.py').write_text(p)
