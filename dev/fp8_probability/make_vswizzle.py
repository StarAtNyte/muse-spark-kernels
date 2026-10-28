from pathlib import Path
import ast
parent=Path('dev/ptx_campaign/attention_fp8p_joint_serial.py').read_text()
node=next(n for n in ast.parse(parent).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_WG_CUDA' for t in n.targets));old=ast.literal_eval(node.value)
s=old.replace('#define WS 2','__device__ __forceinline__ int v64_off(int row,int chunk){int a=row*64+chunk*16;return a^((a>>3)&0x30);}\n#define WS 2')
s=s.replace('vt+core_off(d,warp)','vt+v64_off(d,warp)').replace('const uint64_t vd0 = mkdesc(vt,128,512);','const uint64_t vd0 = mkdesc(vt,16,512)|(2ull<<62);').replace('wg_pv8(o,pp[1],vd0+16)','wg_pv8(o,pp[1],vd0+2)')
lines=parent.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['_WG_CUDA = '+repr(s)+'\n'];out=''.join(lines).replace('muse_wg5_fp8p_joint_serial','muse_wg5_fp8p_vsw64')
Path('dev/ptx_campaign/attention_fp8p_vsw64.py').write_text(out)
Path('dev/ptx_campaign/attention_fp8p_vsw64_bundle.py').write_text(out+'\n_FP8P_SERIAL_CUDA='+repr(old)+'\n')
Path('dev/fp8_probability/vsw64_kernel.cu').write_text(s)
p=Path('dev/modal/drv_fp8p_joint_compare.py').read_text().replace("('stx2',candidate._WG_CUDA),('serial',candidate._FP8P_SERIAL_CUDA)","('vsw64',candidate._WG_CUDA),('original',candidate._FP8P_SERIAL_CUDA)")
p=p.replace('for batch in (8,16,32):','for batch in (16,32):').replace('for seed in ((113,) if batch==8 else (113,7919,3613)):','for seed in (7919,):').replace("'muse_joint_'+name","'muse_joint_vsw_'+name")
Path('dev/modal/drv_fp8p_vsw64.py').write_text(p)
