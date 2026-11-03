import os
import json

experiments_dir = "experiments"
exp_list = sorted(os.listdir(experiments_dir))

public_bests = {
    "muse_bf16_proj": 1.5395,
    "muse_attention_decode": 2.6177,
    "muse_moe_w4a16": 2.3820,
    "muse_decode_layer": 2.2603,
    "muse_prefill_layer": 1.7874
}

all_checks = []
unsubmitted_beats = []
all_screens = []

for exp in exp_list:
    p = os.path.join(experiments_dir, exp)
    if not os.path.isdir(p):
        continue
    
    status_file = os.path.join(p, "status.json")
    manifest_file = os.path.join(p, "manifest.json")
    
    # Check if there is a screen log or analysis
    has_status = os.path.exists(status_file)
    has_analysis = os.path.exists(os.path.join(p, "analysis.md"))
    
    # List files to see if there are screen logs
    files = os.listdir(p)
    screen_logs = [f for f in files if "screen" in f or "paired" in f or "modal" in f or f.endswith(".log")]
    
    if not has_status and (screen_logs or has_analysis):
        all_screens.append((exp, screen_logs))
    
    if has_status:
        try:
            with open(status_file) as f:
                d = json.load(f)
            board = d.get("board")
            mode = d.get("mode")
            job_id = d.get("id")
            result = d.get("result", {})
            passed = result.get("passed", False)
            speedup = result.get("speedup", 0.0)
            ci = result.get("speedup_ci", [])
            
            if mode == "check" and passed:
                all_checks.append((exp, board, speedup, job_id, ci))
                if board in public_bests and speedup > public_bests[board]:
                    unsubmitted_beats.append((exp, board, speedup, public_bests[board], job_id, ci))
        except Exception as e:
            pass

print(f"Total passing check runs found: {len(all_checks)}")
print(f"Total unsubmitted checks beating current public best: {len(unsubmitted_beats)}")
for u in unsubmitted_beats:
    print(f"  {u[0]}: board={u[1]}, check_speedup={u[2]:.5f} > pub_best={u[3]:.4f}, job={u[4]}, CI={u[5]}")

print(f"\nTotal screen-only experiments (never officially checked): {len(all_screens)}")
