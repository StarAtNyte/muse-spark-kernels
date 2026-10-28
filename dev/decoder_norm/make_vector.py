from pathlib import Path
import runpy
m=runpy.run_path('dev/decoder_norm/probe.py');s=m['CUDA']
s=s.replace('const bf16* X, const bf16* W, bf16* Y, int M)', 'const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ Y, int M)')
s=s.replace('for(int col=ni;col<4096;col+=128){float v=__bfloat162float(X[nr*4096+col]);sq=fmaf(v,v,sq);}',r'''
+#pragma unroll
+    for(int q=0;q<4;q++){
+      uint4 pack=*reinterpret_cast<const uint4*>(X+nr*4096+ni*8+q*1024);
+      const bf162* p=reinterpret_cast<const bf162*>(&pack);
+#pragma unroll
+      for(int j=0;j<4;j++){float2 v=__bfloat1622float2(p[j]);sq=fmaf(v.x,v.x,sq);sq=fmaf(v.y,v.y,sq);}
+    }'''.replace('\n+','\n'))
old='''    for(int col=ni;col<4096;col+=128){
      float v=__bfloat162float(X[nr*4096+col]);
      float w=__bfloat162float(W[col]);
      Y[nr*4096+col]=__float2bfloat16_rn((v*inv)*w);
    }'''
new=r'''
#pragma unroll
    for(int q=0;q<4;q++){
      int col=ni*8+q*1024;
      uint4 xp=*reinterpret_cast<const uint4*>(X+nr*4096+col);
      uint4 wp=*reinterpret_cast<const uint4*>(W+col),out;
      const bf162* a=reinterpret_cast<const bf162*>(&xp);
      const bf162* w=reinterpret_cast<const bf162*>(&wp);
      bf162* y=reinterpret_cast<bf162*>(&out);
#pragma unroll
      for(int j=0;j<4;j++){
        float2 v=__bfloat1622float2(a[j]),wt=__bfloat1622float2(w[j]);
        y[j]=__floats2bfloat162_rn((v.x*inv)*wt.x,(v.y*inv)*wt.y);
      }
      *reinterpret_cast<uint4*>(Y+nr*4096+col)=out;
    }'''
assert old in s;s=s.replace(old,new)
Path('dev/decoder_norm/vector.cu').write_text(s)
Path('dev/decoder_norm/vector_probe.py').write_text('CUDA='+repr(s)+'\nNORM='+repr(m['NORM'])+'\n')
