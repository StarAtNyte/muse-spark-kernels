cd /w/run && python -c "import sys; sys.path.insert(0,'/w/wg3'); import wg3_cuda as W; open('k.cu','w').write(W.CUDA_SRC)"
try() { nvcc -arch=sm_90a -O3 -cubin -o t.cubin $1 >/dev/null 2>&1; echo "$2=$?"; }
OLD='if (tw < 128) \*reinterpret_cast<uint4\*>(vt + (tw >> 3) \* 1152 + 1024 + (tw \& 7) \* 16) = make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u);'
sed "s/$OLD/if (tw < 128) { uint32_t* pp = reinterpret_cast<uint32_t*>(vt + (tw >> 3) * 1152 + 1024 + (tw \& 7) * 16); pp[0] = 0x3C003C00u; pp[1] = 0x3C003C00u; pp[2] = 0x3C003C00u; pp[3] = 0x3C003C00u; }/" k.cu > a.cu; grep -c "pp\[3\]" a.cu; try a.cu u32stores
sed "s/$OLD/{ const uint32_t ad = (uint32_t)__cvta_generic_to_shared(vt + (tw >> 3) * 1152 + 1024 + (tw \& 7) * 16); asm volatile(\"st.shared.v4.u32 [%0], {%1,%2,%3,%4};\" :: \"r\"(ad), \"r\"(0x3C003C00u), \"r\"(0x3C003C00u), \"r\"(0x3C003C00u), \"r\"(0x3C003C00u)); }/" k.cu > b.cu; grep -c "st.shared.v4" b.cu; try b.cu asmst
