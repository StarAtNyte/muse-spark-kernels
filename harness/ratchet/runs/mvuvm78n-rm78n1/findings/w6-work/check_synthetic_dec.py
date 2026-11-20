import json

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

# Let's inspect taroigarashi2001 and ymiled and all 15 non-decoder teams
for r in overall:
    t = r["team"]
    if t not in board_data["muse_decode_layer"]:
        # Find where this team sits between two decoder teams
        idx = r["rank"] - 1
        prev_team = overall[idx - 1]["team"] if idx > 0 else None
        next_team = overall[idx + 1]["team"] if idx + 1 < len(overall) else None
        
        prev_dec_val = board_data["muse_decode_layer"].get(prev_team, {}).get("value")
        next_dec_val = board_data["muse_decode_layer"].get(next_team, {}).get("value")
        
        # What is t's sum of values across the other 3 boards?
        p_val = board_data["muse_bf16_proj"].get(t, {}).get("value")
        a_val = board_data["muse_attention_decode"].get(t, {}).get("value")
        m_val = board_data["muse_moe_w4a16"].get(t, {}).get("value")
        
        print(f"OV {r['rank']} {t} (solved {r['solved']}):")
        print(f"   prev {prev_team}: dec_val={prev_dec_val}")
        print(f"   curr values: proj={p_val}, attn={a_val}, moe={m_val}")
        print(f"   next {next_team}: dec_val={next_dec_val}")
        if r["rank"] > 400:
            break
