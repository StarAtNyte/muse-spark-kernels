"""Lossless bitwise codec round trip for generated weights and special BF16 patterns."""
import ctypes as C, torch, math, sys, json, warnings
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
'unweight_decode_launch':([P,P,P,P,P,P,P,P,Z,Z,Z,U,Z,P],C.c_int),
'unweight_encode_to_palette_launch':([P,P,P,P,P,P,Z,Z,Z,P],C.c_int),
'unweight_reconstructive_matmul_launch':([P,P,Z,Z,Z,U,P,P,P,P,P,Z,Z,C.c_bool,U,P],C.c_int)}
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
 palette_idx=alloc(n//2);vpal=alloc(max(64,verbatim*64));idx.zero_()
 call('unweight_encode_to_palette_launch',ptr(w),ptr(inverse),ptr(meta),ptr(palette_idx),ptr(vpal),ptr(idx),rows,cols,132,stream)
 return sm,palette_idx,meta,vpal,palette,int(maxverb.item()),verbatim
weights=R.generate_weights(R.WEIGHTS_SEED)
w=weights['pooler']; sm,pal,meta,vpal,palette,maxverb,nverb=run(w,'pooler')
flush=torch.empty(256*1024*1024//4,dtype=torch.int32,device='cuda')
def cap(fn):
 out=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with warnings.catch_warnings(record=True) as seen:
  warnings.simplefilter("always")
  with torch.cuda.graph(g):out=fn()
 if any("graph is empty" in str(z.message).lower() for z in seen):raise RuntimeError("empty CUDA graph")
 g.replay();torch.cuda.synchronize();return g,out
def timing(g):
 flush.add_(1);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
import statistics
for M in (128,256):
 data=R.generate_input(M,'pooler',9000+M);x=data['x']; n,k=w.shape;stream=P(torch.cuda.current_stream().cuda_stream)
 bg,bo=cap(lambda:x@w.T)
 ok,msg=R.check_implementation(data,bo,weights);print('CHECK',M,'torch',ok,str(msg)[:160],flush=True)
 if not ok:raise RuntimeError('baseline failed')
 for v in (0,1,2,4,5,6,7):
  o=torch.empty((M,n),dtype=torch.bfloat16,device='cuda')
  def fn():
   rc=lib.unweight_reconstructive_matmul_launch(ptr(x),ptr(o),M,n,k,0,ptr(sm),ptr(pal),ptr(meta),ptr(vpal),ptr(palette),maxverb,nverb,True,v,P(torch.cuda.current_stream().cuda_stream))
   if rc:raise RuntimeError(f'variant {v}: rc {rc}')
   return o
  try:
   g,out=cap(fn);ok,msg=R.check_implementation(data,out,weights)
   print('CHECK',M,v,ok,str(msg)[:160],flush=True)
   if not ok:continue
   bt=[];ct=[]
   for i in range(16):
    if i&1:ct.append(timing(g));bt.append(timing(bg))
    else:bt.append(timing(bg));ct.append(timing(g))
   a=statistics.median(bt);b=statistics.median(ct)
   print('TIMING',json.dumps(dict(m=M,variant=v,base_us=a,candidate_us=b,ratio=a/b)),flush=True)
  except Exception as e:
   print('VARIANT_ERROR',M,v,type(e).__name__,str(e)[:200],flush=True);raise
print('DONE',flush=True)
