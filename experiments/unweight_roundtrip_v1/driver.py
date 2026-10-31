"""Lossless bitwise codec round trip for generated weights and special BF16 patterns."""
import ctypes as C, torch, math, sys, json
sys.path.insert(0,'/w/kernels/muse_bf16_proj')
import reference as R
lib=C.CDLL('/opt/libunweight.so')
P=C.c_void_p; Z=C.c_size_t; U=C.c_uint32; B=C.c_uint8
funcs={'unweight_collect_stats_and_extract_sm_launch':([P,P,P,Z,Z,P],None),
'unweight_build_encode_table_and_palette_launch':([P,P,P,P,P,P],None),
'unweight_build_decode_table_launch':([P,P,P,B,P],None),
'unweight_compute_tile_metadata_launch':([P,P,P,P,P,Z,Z,P],None),
'unweight_metadata_lens_to_offsets_launch':([P,Z,P,P,P],None),
'unweight_encode_launch':([P,P,P,P,P,P,P,Z,Z,P],None),
'unweight_decode_launch':([P,P,P,P,P,P,P,P,Z,Z,Z,U,Z,P],C.c_int)}
for k,(args,ret) in funcs.items():
 f=getattr(lib,k);f.argtypes=args;f.restype=ret
ptr=lambda t:P(t.data_ptr())
def call(name,*args):
 ret=getattr(lib,name)(*args)
 if ret not in (None,0):raise RuntimeError(f'{name} returned CUDA code {ret}')
 torch.cuda.synchronize()
def run(w,label):
 w=w.contiguous();rows,cols=w.shape;assert rows%64==0 and cols%64==0
 n=rows*cols;tiles=n//4096;batch=((tiles+63)//64)*64;device=w.device; stream=P(torch.cuda.current_stream().cuda_stream)
 def alloc(num,dtype=torch.uint8):return torch.zeros(num,dtype=dtype,device=device)
 sm=alloc(n);hist=alloc(256,torch.int32);enc_table=alloc(257*4);palette=alloc(16);inverse=torch.full((256,),255,dtype=torch.uint8,device=device);dsize=alloc(1,torch.int32)
 offsets=alloc(batch*64,torch.int16);meta=alloc(batch*4,torch.int32);maxverb=alloc(1,torch.int32)
 call('unweight_collect_stats_and_extract_sm_launch',ptr(w),ptr(sm),ptr(hist),rows,cols,stream)
 call('unweight_build_encode_table_and_palette_launch',ptr(hist),ptr(enc_table),ptr(palette),ptr(inverse),ptr(dsize),stream)
 size=int(dsize.item());assert size>0 and size<=32768 and not size&(size-1),size
 dec=alloc(size*4);call('unweight_build_decode_table_launch',ptr(enc_table),ptr(inverse),ptr(dec),int(math.log2(size)),stream)
 call('unweight_compute_tile_metadata_launch',ptr(w),ptr(offsets),ptr(meta),ptr(enc_table),ptr(maxverb),rows,cols,stream)
 clen=alloc(1,torch.int32);vcount=alloc(1,torch.int32)
 call('unweight_metadata_lens_to_offsets_launch',ptr(meta),tiles,ptr(clen),ptr(vcount),stream)
 words=int(clen.item());verbatim=int(vcount.item()); assert 0<=words<=n and 0<=verbatim<=n//64
 encoded=alloc(max(4,words*4));vbuf=alloc(max(64,verbatim*64));idx=alloc(1,torch.int32)
 call('unweight_encode_launch',ptr(w),ptr(meta),ptr(offsets),ptr(enc_table),ptr(encoded),ptr(vbuf),ptr(idx),rows,cols,stream)
 out=torch.empty_like(w);idx.zero_();call('unweight_decode_launch',ptr(sm),ptr(out),ptr(offsets),ptr(meta),ptr(vbuf),ptr(encoded),ptr(dec),ptr(idx),rows,cols,132,1,size,stream)
 a=w.view(torch.int16);b=out.view(torch.int16);errors=int((a!=b).sum().item())
 print('ROUNDTRIP',json.dumps(dict(label=label,shape=[rows,cols],errors=errors,encoded_bytes=words*4,sm_bytes=n,verbatim_bytes=verbatim*64,metadata_bytes=tiles*16,offset_bytes=tiles*128,max_verbatim_per_tile=int(maxverb.item()),decode_table_entries=size)),flush=True)
 if errors:raise AssertionError(f'{label}: {errors} bit mismatches')
weights=R.generate_weights(R.WEIGHTS_SEED)
for name in ('qkv_gate','pooler'):run(weights[name],name)
g=torch.Generator(device='cuda').manual_seed(7919);x=torch.randn((512,512),device='cuda',generator=g).to(torch.bfloat16)
patterns=torch.tensor([0,32768,1,127,128,32640,65408,32704,65535],device='cuda',dtype=torch.int32).to(torch.int16)
x.view(torch.int16).flatten()[:patterns.numel()]=patterns
run(x,'special_bits')
print('DONE',flush=True)
