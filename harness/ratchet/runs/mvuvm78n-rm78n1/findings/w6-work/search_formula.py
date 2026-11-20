import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]

# baseline times:
# base_proj = 4830.7268
# base_attn = 3766.3305
# base_moe = 13448.5393
# base_dec = 22620.2471

# For taroigarashi2001:
# p_val = 3146.6833
# a_val = 1334.8217
# m_val = 5442.0412
# What combination of p_val, a_val, m_val gives between 12501.65 and 12524.03?
# Note: In a decoder layer, there are also norms and residuals.
# What is the baseline non-component overhead in decoder?
# overhead = base_dec - (base_proj + base_attn + base_moe) = 22620.2471 - (4830.7268 + 3766.3305 + 13448.5393) = 22620.2471 - 22045.5966 = 574.6505
# If taroigarashi2001 had p_val + a_val + m_val + overhead = 9923.5462 + 574.6505 = 10498.20 -> that would be rank 150!
# What about a weighted sum, or what about speedup?
# What if missing problems get stock baseline (speedup=1.0, value=base)?
# What if overall rank is:
# Wait! Let's check what value or speedup is used!

# Let's search for any formula F(p, a, m, d) that puts taroigarashi2001 at rank 366 and ymiled at 385!
with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

# Let's inspect all teams between 360 and 390
for r in overall[360:390]:
    t = r["team"]
    row = board_data["muse_decode_layer"].get(t)
    if row:
        print(f"OV {r['rank']:3d} {t:18s} DEC: speedup={row['speedup']:.4f} val={row['value']:.2f}")
    else:
        p_row = board_data["muse_bf16_proj"].get(t, {})
        a_row = board_data["muse_attention_decode"].get(t, {})
        m_row = board_data["muse_moe_w4a16"].get(t, {})
        print(f"OV {r['rank']:3d} {t:18s} NO DEC! P={p_row.get('speedup')} A={a_row.get('speedup')} M={m_row.get('speedup')}")
