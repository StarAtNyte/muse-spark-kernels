import json
import math

boards = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]

board_data = {}
for b in boards:
    with open(f"/tmp/board_{b}.json") as f:
        board_data[b] = {r["team"]: r for r in json.load(f)["rows"]}

with open("/tmp/live_overall.json") as f:
    overall = json.load(f)["rows"]

# Let's inspect the first 100 rows of overall, and see how their ranks/speedups relate.
# Notice earlier:
# Rank 1: mpeng19 ranks=[26, 23, 4, 1]
# Rank 2: pjmullin3 ranks=[22, 4, 2, 2]
# Rank 3: colinlikescode ranks=[2, 6, 6, 3]
# Wait, why was mpeng19 rank 1 when pjmullin3 had ranks [22, 4, 2, 2] and colinlikescode had [2, 6, 6, 3]?
# Wait! In the printout earlier:
# Rank  1: mpeng19 ranks=[26, 23, 4, 1]
# Rank  2: pjmullin3 ranks=[22, 4, 2, 2]
# Rank  3: colinlikescode ranks=[2, 6, 6, 3]
# Rank  4: abhishekgahlot2 ranks=[39, 12, 20, 4]
# Rank  5: dolanzhao ranks=[None, None, 1, 5]
# Rank  6: tonybolivar ranks=[573, 218, 41, 6]
# Wait! Notice decoder rank:
# mpeng19: decoder rank 1
# pjmullin3: decoder rank 2
# colinlikescode: decoder rank 3
# abhishekgahlot2: decoder rank 4
# dolanzhao: decoder rank 5
# tonybolivar: decoder rank 6
# PierreNowak: decoder rank 7
# Aspect5: decoder rank 8
# Abarbir12: decoder rank 9
# ntovas: decoder rank 10
# mgpai22: decoder rank 11
# reformedot: decoder rank 12
# olety: decoder rank 13
# zzz0906: decoder rank 14
# bomboclatdaru: decoder rank 15
# apullin: decoder rank 16
# nhat-nguyen: decoder rank 17
# zereraz: decoder rank 18
# concept-guest: decoder rank 19
# caoshiyi: decoder rank 20
# reyanshsolis: decoder rank 21
# tie-pilot-qxw: decoder rank 22
# AkshitPareek: decoder rank 23
# picsoulabs: decoder rank 24
# ZhimaoL: decoder rank 25
# Lurrobert: decoder rank 26
# keell0renz: decoder rank 27
# StarAtNyte: decoder rank 28

print("Let's check decoder ranks for all overall rows 1 to 365:")
all_match = True
for r in overall[:365]:
    team = r["team"]
    dec_rank = board_data["muse_decode_layer"].get(team, {}).get("rank")
    if dec_rank != r["rank"]:
        print(f"Mismatch at overall {r['rank']}: team {team}, dec_rank {dec_rank}")
        all_match = False
print("All match 1..365?", all_match)
