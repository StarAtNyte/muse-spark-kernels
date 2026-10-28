from pathlib import Path
src=Path('kernels/muse_decode_layer/submission.py').read_text()
changes={
"(128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 3, 35, 0, 5])":"(128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5])",
"(256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])":"(256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 4, 35, 0, 3])",
}
for a,b in changes.items():
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
out=Path('dev/decoder_projection_stack/submission_combined.py');out.write_text(src)
print(out,len(src),'bytes')
