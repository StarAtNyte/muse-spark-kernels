@triton.jit
def _decode_words(words, scales):
    a,b,c,d,e,f,g,h = tl.inline_asm_elementwise(
        r"""{
.reg .b32 packed, q, mag, bits, tmp, sign;
.reg .b16 h, scale;
.reg .pred small;
mov.b32 packed, $8;
mov.b16 scale, $9;
shr.u32 q, packed, 0;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $0, h, scale;
shr.u32 q, packed, 16;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $1, h, scale;
shr.u32 q, packed, 8;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $2, h, scale;
shr.u32 q, packed, 24;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $3, h, scale;
shr.u32 q, packed, 4;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $4, h, scale;
shr.u32 q, packed, 20;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $5, h, scale;
shr.u32 q, packed, 12;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $6, h, scale;
shr.u32 q, packed, 28;
and.b32 mag, q, 7;
shl.b32 bits, mag, 6;
add.u32 bits, bits, 16128;
mul.lo.u32 tmp, mag, 16128;
setp.lt.u32 small, mag, 2;
selp.b32 bits, tmp, bits, small;
and.b32 sign, q, 8;
shl.b32 sign, sign, 12;
or.b32 bits, bits, sign;
cvt.u16.u32 h, bits;
mul.rn.bf16 $7, h, scale;
}""",
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[words, scales], dtype=(tl.bfloat16,)*8, is_pure=True, pack=1)
    p0, p1 = tl.join(a,b), tl.join(c,d)
    p2, p3 = tl.join(e,f), tl.join(g,h)
    return tl.join(tl.join(p0,p1),tl.join(p2,p3)).reshape(words.shape[0], words.shape[1]*8)


