# Let's test combinations of (proj, attn, moe) for taroigarashi2001 and ymiled
# taroigarashi2001:
# proj_speedup = 1.5352 (value = 3146.68)
# attn_speedup = 2.8216 (value = 1334.82)
# moe_speedup  = 2.4712 (value = 5442.04)
# Target dec_speedup: ~1.8078 (effective value ~12513 us)
#
# Baseline values:
# base_proj = 4830.73 us
# base_attn = 3766.33 us
# base_moe  = 13448.54 us
# base_dec  = 22620.25 us
# Note: base_proj + base_attn + base_moe = 22045.6 us (~97.5% of base_dec).
#
# What if the virtual decoder time is computed assuming stock/unoptimized for missing parts or what?
# What if virtual_time = (base_dec - base_proj) + proj_val? No.
# What if virtual_time = val_proj + val_attn + val_moe + overhead?
# Earlier we found val_proj + val_attn + val_moe = 3146.68 + 1334.82 + 5442.04 = 9923.54 us.
# 22620.25 / 9923.54 = 2.279x! That would be rank ~22, not 366!
# Why is taroigarashi2001 at rank 366 (speedup ~1.808x)?
# What if:
# What if harmonic mean, geometric mean, or what?
import math

sp_p = 1.5352
sp_a = 2.8216
sp_m = 2.4712

# Geometric mean:
print("Geomean of (P, A, M):", (sp_p * sp_a * sp_m)**(1/3)) # 2.203

# Geomean of (P, A, M, 1.0):
print("Geomean of (P, A, M, 1.0):", (sp_p * sp_a * sp_m * 1.0)**(1/4)) # 1.829

# Weighted geomean?
# What about ymiled?
ym_p = 1.5534
ym_a = 2.3461
ym_m = 2.4652
print("ymiled Geomean of (P, A, M, 1.0):", (ym_p * ym_a * ym_m * 1.0)**(1/4)) # 1.732

# What if arithmetic mean with 1.0?
print("taroigarashi arith with 1.0:", (sp_p + sp_a + sp_m + 1.0) / 4) # 1.957

# What if: In decoder layer, what are the weights of the loads or components?
# In decoder layer, the baseline time is 22620.25.
# If taroigarashi only submitted 3 problems, but has NO submission for decoder layer:
# What if:
# In overall leaderboard, is overall leaderboard:
# Wait! What if overall leaderboard is:
# Let's test all 1680 rows!
