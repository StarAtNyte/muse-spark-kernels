import os
import json

experiments_dir = "experiments"
exp_list = sorted(os.listdir(experiments_dir))

records = []
for exp in exp_list:
    p = os.path.join(experiments_dir, exp)
    if not os.path.isdir(p):
        continue
    status_file = os.path.join(p, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file) as f:
                d = json.load(f)
            board = d.get("board")
            mode = d.get("mode")
            job_id = d.get("id")
            res = d.get("result", {})
            if res.get("passed"):
                records.append({
                    "exp": exp,
                    "board": board,
                    "mode": mode,
                    "job_id": job_id,
                    "speedup": res.get("speedup"),
                    "ci": res.get("speedup_ci"),
                    "score_us": res.get("score_us"),
                    "baseline_us": res.get("baseline_us")
                })
        except:
            pass

# Let's inspect highest checks per board
for b in ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]:
    print(f"\n=== Top 5 CHECK runs for {b} ===")
    checks = [r for r in records if r["board"] == b and r["mode"] == "check"]
    checks.sort(key=lambda x: x["speedup"] or 0, reverse=True)
    for c in checks[:5]:
        print(f"  {c['exp']}: speedup={c['speedup']:.5f}, job={c['job_id']}, CI={c['ci']}")

    print(f"=== Top 3 SCORED runs for {b} ===")
    scored = [r for r in records if r["board"] == b and r["mode"] == "run"]
    scored.sort(key=lambda x: x["speedup"] or 0, reverse=True)
    for s in scored[:3]:
        print(f"  {s['exp']}: speedup={s['speedup']:.5f}, job={s['job_id']}, CI={s['ci']}")
