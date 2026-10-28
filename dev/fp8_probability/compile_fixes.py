from pathlib import Path
s=Path('dev/fp8_probability/joint_kernel.cu').read_text()
q='''    wg_fence();
    wg_qk_first(sacc, qd0, kdesc);
    wg_qk_acc(sacc, qd1, kdesc + 16);
    wg_commit();
'''
assert s.count(q)==1
serial=s.replace(q,'').replace('    wg_wait0();',q+'    wg_wait0();')
Path('dev/fp8_probability/joint_serial.cu').write_text(serial)
old='''      int matrix=lane>>3,d=pr*32+(matrix>>1)*16+2*(lane&7)+(matrix&1);
      uint32_t ad=__cvta_generic_to_shared(vt+core_off(d,warp));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"::"r"(ad),"r"(z[0]),"r"(z[1]),"r"(z[2]),"r"(z[3]):"memory");'''
new='''      int d=pr*32+2*(lane&7)+((lane>>3)&1);
      uint32_t ad0=__cvta_generic_to_shared(vt+core_off(d,warp));
      uint32_t ad1=__cvta_generic_to_shared(vt+core_off(d+16,warp));
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0],{%1,%2};"::"r"(ad0),"r"(z[0]),"r"(z[1]):"memory");
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0],{%1,%2};"::"r"(ad1),"r"(z[2]),"r"(z[3]):"memory");'''
assert s.count(old)==1
Path('dev/fp8_probability/joint_stx2.cu').write_text(s.replace(old,new))
Path('dev/fp8_probability/joint_serial_stx2.cu').write_text(serial.replace(old,new))
