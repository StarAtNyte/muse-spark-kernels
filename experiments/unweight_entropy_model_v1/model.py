"""CPU-only Gaussian BF16 exponent statistics; model, not exact judge RNG or actual codec."""
import collections, math, random, struct, json
rng=random.Random(39031); counts=collections.Counter(); n=500_000
for _ in range(n):
    x=rng.gauss(0,1)/math.sqrt(8192)
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    bf=(bits+0x7fff+((bits>>16)&1))>>16
    counts[(bf>>7)&255]+=1
entropy=-sum((v/n)*math.log2(v/n) for v in counts.values())
top=counts.most_common(16); miss=1-sum(v for _,v in top)/n
out={'samples':n,'seed':39031,'entropy_bits':entropy,'top16_escape_fraction':miss,'ideal_exponent_plus_sign_mantissa_bytes_per_weight':(8+entropy)/8,'fixed_4bit_exponent_plus_sign_mantissa_bytes_per_weight':1.5,'top_exponents':top,'independent_64_value_row_fallback_probability_model':1-(1-miss)**64}
print(json.dumps(out,indent=2))
