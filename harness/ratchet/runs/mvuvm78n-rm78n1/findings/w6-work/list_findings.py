import os

experiments_dir = "experiments"
exp_list = sorted(os.listdir(experiments_dir))

for exp in exp_list:
    p = os.path.join(experiments_dir, exp)
    if not os.path.isdir(p):
        continue
    f_path = os.path.join(p, "FINDINGS.md")
    a_path = os.path.join(p, "analysis.md")
    status_path = os.path.join(p, "status.json")
    
    if os.path.exists(f_path):
        with open(f_path) as f:
            first_lines = "".join(f.readlines()[:4]).replace("\n", " ")
        print(f"[{exp}] (has_status={os.path.exists(status_path)}): {first_lines[:150]}")
    elif os.path.exists(a_path) and not os.path.exists(status_path):
        with open(a_path) as f:
            first_lines = "".join(f.readlines()[:4]).replace("\n", " ")
        print(f"[{exp}] (NO STATUS): {first_lines[:150]}")
