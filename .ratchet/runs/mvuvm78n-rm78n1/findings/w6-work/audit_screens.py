import os
import json

experiments_dir = "experiments"
exp_list = sorted(os.listdir(experiments_dir))

all_screens = []

for exp in exp_list:
    p = os.path.join(experiments_dir, exp)
    if not os.path.isdir(p):
        continue
    
    status_file = os.path.join(p, "status.json")
    has_status = os.path.exists(status_file)
    has_analysis = os.path.exists(os.path.join(p, "analysis.md"))
    
    files = os.listdir(p)
    screen_logs = [f for f in files if "screen" in f or "paired" in f or "modal" in f or f.endswith(".log") or f.endswith(".md") or f.endswith(".txt")]
    
    if not has_status and (screen_logs or has_analysis):
        all_screens.append((exp, screen_logs))

all_screens_info = []

for exp, logs in all_screens:
    p = os.path.join(experiments_dir, exp)
    summary = ""
    analysis_path = os.path.join(p, "analysis.md")
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            lines = f.readlines()[:15]
            summary = "".join(lines)
    else:
        for l in logs:
            lp = os.path.join(p, l)
            try:
                with open(lp) as f:
                    content = f.read()
                    if any(w in content.lower() for w in ["speedup", "ratio", "win", "faster", "better", "gain"]):
                        summary += f"[{l}]: " + content[:250].replace("\n", " ") + "\n"
            except:
                pass
    if summary:
        all_screens_info.append((exp, summary))

print(f"Screens with interesting notes/logs: {len(all_screens_info)}")
for exp, s in all_screens_info[:25]:
    print(f"\n=== {exp} ===")
    print(s[:350])
