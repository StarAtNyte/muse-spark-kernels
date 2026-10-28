from pathlib import Path
s=Path('dev/transpose_verify/kernel.cu').read_text()
s=s.replace('ST*4096+4096','ST*8192+8192').replace('(tile%ST)*4096','(tile%ST)*8192').replace('mem+ST*4096','mem+ST*8192')
s=s.replace('(n+15)/16','(n+31)/32').replace('i<128;i+=512','i<256;i+=512').replace('li=tile*16+key','li=tile*32+key').replace('dst+2048','dst+4096')
s=s.replace('if(tid<128)','if(tid<256)').replace('raw+2048','raw+4096')
a=s.index('  const uint8_t* raw=');z=s.index('  uint8_t* vf=',a)
s=s[:a]+'''
  const uint8_t* raw=mem+(tile%ST)*8192;float s[2][4]={};
#pragma unroll
  for(int nt=0;nt<2;nt++){
#pragma unroll
   for(int k=0;k<2;k++){uint32_t a[4];ld(a,raw+off(nt*16+(lane&7)+8*((lane>>3)&1),h*64+k*32+16*(lane>>4)));qmma(s[nt],a,qb[k][0],qb[k][1]);}
#pragma unroll
   for(int r=0;r<4;r++)s[nt][r]=(tile*32+nt*16+g+(r>=2?8:0)<n && begin+tile*32+nt*16+g+(r>=2?8:0)<=prefix+qt)?s[nt][r]*scale:-INFINITY;
  }
'''+s[z:]
s=s.replace('float mx[2]={fmaxf(s[0],s[2]),fmaxf(s[1],s[3])};','float mx[2]={fmaxf(fmaxf(s[0][0],s[0][2]),fmaxf(s[1][0],s[1][2])),fmaxf(fmaxf(s[0][1],s[0][3]),fmaxf(s[1][1],s[1][3]))};')
a=s.index('  float p[4];');z=s.index('  __syncthreads();\n }',a)
s=s[:a]+'''
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
  __syncthreads();
#pragma unroll
  for(int nt=0;nt<2;nt++){
   __half2 hp0=__floats2half2_rn(p[nt][0],p[nt][1]),hp8=__floats2half2_rn(p[nt][2],p[nt][3]);
   uint32_t p0=*reinterpret_cast<uint32_t*>(&hp0),p8=*reinterpret_cast<uint32_t*>(&hp8),z[2];
   int sa=8*t+(g>>1),sb=sa+4,sh=(g&1)*16;
   z[0]=((__shfl_sync(0xffffffff,p0,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p0,sb)>>sh)<<16);
   z[1]=((__shfl_sync(0xffffffff,p8,sa)>>sh)&65535)|((__shfl_sync(0xffffffff,p8,sb)>>sh)<<16);
#pragma unroll
   for(int pr=0;pr<4;pr++){uint32_t r[4];ldt(r,vf+voff(nt*16+(lane&7)+8*((lane>>3)&1),h*128+pr*32+16*(lane>>4)));
    uint32_t a[4]={r[0],r[2],r[1],r[3]};pmma(o[pr],a,z);
   }
  }
'''+s[z:]
s=s.replace('3*4096+4096+','3*8192+8192+')
Path('dev/transpose_verify/kernel_tile32.cu').write_text(s)
