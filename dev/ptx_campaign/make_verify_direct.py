from pathlib import Path
s=Path('kernels/muse_attention_decode/submission.py').read_text()
a=s.index('  for (int i = tid; i < n; i += 256) {')
b=s.index('  for (int c = tw;',a)
s=s[:a]+s[b:]
old='''        const long rr = li < n ? rows[li] : 0;
        asm volatile("cp.async.cg.shared.global'''
new='''        const int tok = begin + li;
        const int pg = tok >> pshift;
        const int phys = (li < n && pg < table_n) ? __ldg(TABLE + b * table_n + pg) : -1;
        const long rr = phys < 0 ? 0 : ((phys << pshift) + (tok & pmask));
        asm volatile("cp.async.cg.shared.global'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace(' + rmax * 4;', ';',1)
s=s.replace('load_inline("muse_wg5_attn",','load_inline("muse_wg5_direct",',1)
Path('dev/ptx_campaign/attention_verify_direct.py').write_text(s)
p=Path('dev/modal/drv_verify_pipeline.py').read_text()
p=p.replace('for stages in (2, 3, 4):','for stages in (2, 3):').replace("f'verify_stage_{stages}'", "f'verify_direct_stage_{stages}'").replace('for splits in (8, 16, 32, 64):','for splits in (8, 16, 32):')
Path('dev/modal/drv_verify_direct.py').write_text(p)
