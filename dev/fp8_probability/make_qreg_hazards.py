from pathlib import Path
import ast
p=Path('dev/ptx_campaign/attention_fp8p_qreg.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));s=ast.literal_eval(n.value)
rw=s.replace(':"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(b));',',"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]):"l"(b));')
assert rw!=s
serial=s.replace('    wg_qk_reg_first(sacc, qa[0], kdesc);','    wg_qk_reg_first(sacc, qa[0], kdesc);\n    wg_commit();wg_wait0();')
Path('dev/ptx_campaign/attention_fp8p_qreg_hazards.py').write_text(p+'\n_QREG_RW_CUDA='+repr(rw)+'\n_QREG_SERIAL_CUDA='+repr(serial)+'\n')
