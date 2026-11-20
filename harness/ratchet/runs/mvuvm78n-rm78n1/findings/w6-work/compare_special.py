import json
import statistics

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

# Let's inspect teams that did NOT solve all 4 boards, or specifically taroigarashi2001, dolanzhao, caoshiyi, SamrendraS, ymiled
special_teams = ["mpeng19", "pjmullin3", "colinlikescode", "abhishekgahlot2", "dolanzhao", "tonybolivar", "caoshiyi", "StarAtNyte", "taroigarashi2001", "ymiled"]

print(f"{'Team':20s} {'OV':4s} {'Solv':4s} {'Proj':5s} {'Attn':5s} {'MoE':5s} {'Dec':5s}")
for t in special_teams:
    ov_row = next((r for r in overall if r["team"] == t), None)
    ov_r = ov_row["rank"] if ov_row else "N/A"
    solv = ov_row["solved"] if ov_row else "N/A"
    p_r = board_data["muse_bf16_proj"].get(t, {}).get("rank", "-")
    a_r = board_data["muse_attention_decode"].get(t, {}).get("rank", "-")
    m_r = board_data["muse_moe_w4a16"].get(t, {}).get("rank", "-")
    d_r = board_data["muse_decode_layer"].get(t, {}).get("rank", "-")
    print(f"{t:20s} {str(ov_r):4s} {str(solv):4s} {str(p_r):5s} {str(a_r):5s} {str(m_r):5s} {str(d_r):5s}")
