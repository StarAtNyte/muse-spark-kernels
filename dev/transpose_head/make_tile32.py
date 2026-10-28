from pathlib import Path
import runpy
old=runpy.run_path('dev/transpose_head/decode.py')
s=old['CUDA']
s=s.replace('ST*8192','ST*16384').replace('(tile%ST)*8192','(tile%ST)*16384').replace('2*8192','2*16384')
s=s.replace('(n+31)/32','(n+63)/64').replace('i<256','i<512').replace('li=tile*32+key','li=tile*64+key').replace('dst+4096','dst+8192')
a=s.index('  const uint8_t* raw=');z=s.index('  float mx[2]=',a)
s=s[:a]+r'''
  const uint8_t* raw=mem+(tile%ST)*16384;float s[2][4]={};
#pragma unroll
  for(int nt=0;nt<2;nt++){
#pragma unroll
   for(int k=0;k<2;k++){uint32_t a[4];ld(a,raw+off(part*32+nt*16+(lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));qmma(s[nt],a,qb[k][0],qb[k][1]);}
#pragma unroll
   for(int r=0;r<4;r++)s[nt][r]=(tile*64+part*32+nt*16+g+(r>=2?8:0)<n)?s[nt][r]*scale:-INFINITY;
  }
'''+s[z:]
s=s.replace('float mx[2]={fmaxf(s[0],s[2]),fmaxf(s[1],s[3])};','float mx[2]={fmaxf(fmaxf(s[0][0],s[0][2]),fmaxf(s[1][0],s[1][2])),fmaxf(fmaxf(s[0][1],s[0][3]),fmaxf(s[1][1],s[1][3]))};')
a=s.index('  float p[4];');z=s.index('  __syncthreads();\n }',a)
s=s[:a]+r'''
  float p[2][4];
#pragma unroll
  for(int nt=0;nt<2;nt++){
#pragma unroll
   for(int r=0;r<4;r++){float safe=mm[r&1]==-INFINITY?0.f:mm[r&1];p[nt][r]=exp2f(s[nt][r]-safe);}
  }
#pragma unroll
  for(int c=0;c<2;c++){float sum=p[0][c]+p[0][c+2]+p[1][c]+p[1][c+2];
#pragma unroll
   for(int x=4;x<=16;x*=2)sum+=__shfl_xor_sync(0xffffffff,sum,x);ll[c]+=sum;}
#pragma unroll
  for(int nt=0;nt<2;nt++){
   __half2 hp0=__floats2half2_rn(p[nt][0],p[nt][1]),hp8=__floats2half2_rn(p[nt][2],p[nt][3]);
   uint32_t p0=*reinterpret_cast<uint32_t*>(&hp0),p8=*reinterpret_cast<uint32_t*>(&hp8),z[2];
   int sa=8*t+(g>>1),sb=sa+4,sh=(g&1)*16;
   z[0]=((__shfl_sync(0xffffffff,p0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>sh)<<16);
   z[1]=((__shfl_sync(0xffffffff,p8,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>sh)<<16);
#pragma unroll
   for(int pr=0;pr<2;pr++){uint32_t r[4];ldt(r,raw+8192+off(part*32+nt*16+(lane&7)+8*((lane>>3)&1),h*64+pr*32+16*(lane>>4)));
#pragma unroll
    for(int odd=0;odd<2;odd++){uint32_t a[4];a[0]=cv2(__byte_perm(r[0],0,odd?0x4431:0x4420));a[1]=cv2(__byte_perm(r[2],0,odd?0x4431:0x4420));a[2]=cv2(__byte_perm(r[1],0,odd?0x4431:0x4420));a[3]=cv2(__byte_perm(r[3],0,odd?0x4431:0x4420));pmma(o[2*pr+odd],a,z);}}
  }
'''+s[z:]
Path('dev/transpose_head/decode_tile32.cu').write_text(s)
Path('dev/transpose_head/decode_tile32.py').write_text('CPP='+repr(old['CPP'])+'\nCUDA='+repr(s)+'\n')
