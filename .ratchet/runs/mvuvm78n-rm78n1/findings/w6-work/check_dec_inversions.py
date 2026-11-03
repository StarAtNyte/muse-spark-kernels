import json

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

with open("/tmp/board_muse_decode_layer.json") as f:
    dec = {r["team"]: r for r in json.load(f)["rows"]}

# Collect all teams in overall that have a dec entry
dec_teams_in_ov = [r["team"] for r in overall if r["team"] in dec]

# Check if their decoder speedups are strictly non-increasing:
inversions = []
for i in range(len(dec_teams_in_ov) - 1):
    t1 = dec_teams_in_ov[i]
    t2 = dec_teams_in_ov[i+1]
    sp1 = dec[t1]["speedup"]
    sp2 = dec[t2]["speedup"]
    val1 = dec[t1]["value"]
    val2 = dec[t2]["value"]
    # We want val1 <= val2 (or sp1 >= sp2)
    if val1 > val2:
        inversions.append((i, t1, t2, val1, val2, sp1, sp2))

print(f"Total decoder teams in overall: {len(dec_teams_in_ov)}")
print(f"Total value inversions: {len(inversions)}")
if inversions:
    print("First 5 inversions:")
    for inv in inversions[:5]:
        print(inv)
