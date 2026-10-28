from pathlib import Path
s=Path('dev/ptx_campaign/attention_verify_vswizzle.py').read_text()
a=s.index('_WG_CUDA ='); b=s.index('_WG_MOD =',a)
x=s[a:b]
repls={
'__launch_bounds__(256, 2)':'__launch_bounds__(128, 4)',
'wgi = tid >> 7':'wgi = 0', 'kh = wgi, b = blockIdx.z':'kh = blockIdx.y, b = blockIdx.z',
'smem + wgi * 4096':'smem', 'smem + 8192 + wgi * 9216':'smem + 4096',
'smem + 8192 + 18432':'smem + 4096 + 9216',
'stages + WS * 16384':'stages + WS * 8192', 'i += 256':'i += 128',
'const int ci = tid & 15;':'const int ci = tid & 7;', 'const int cc = ci & 7;':'const int cc = ci & 3;',
'const bool isv = ci >= 8;':'const bool isv = ci >= 4;',
'(isv ? VC : KC) + cc * 16':'(isv ? VC : KC) + kh * 64 + cc * 16',
'(tid >> 4) + 16 * i':'(tid >> 3) + 16 * i',
'(8192 + key * 128 + (cc ^ (key & 7)) * 16) : ((cc >> 2) * 4096 + core_off(key, cc & 3))':'(4096 + key * 64 + (cc ^ (key & 3)) * 16) : core_off(key, cc)',
'stage * 16384':'stage * 8192', 'stages + wgi * 4096':'stages',
'vkey * 128 + ((wgi * 4 + cs) ^ (vkey & 7)) * 16':'vkey * 64 + (cs ^ (vkey & 3)) * 16',
'rd_stage * 8192 + 8192':'rd_stage * 8192 + 4096',
'(rd_stage * 1024)':'(rd_stage * 512)',
'1024 + 8192 + 18432 + (size_t)WS * 16384':'1024 + 4096 + 9216 + (size_t)WS * 8192',
'dim3 grid(splits, 1, B);':'dim3 grid(splits, 2, B);',
'<<<grid, 256, dyn, st>>>':'<<<grid, 128, dyn, st>>>'}
for old,new in repls.items():
 assert old in x,old
 x=x.replace(old,new)
s=s[:a]+x+s[b:]
s=s.replace('load_inline("muse_wg5_vswizzle",','load_inline("muse_wg5_onehead",')
Path('dev/ptx_campaign/attention_verify_onehead.py').write_text(s)
