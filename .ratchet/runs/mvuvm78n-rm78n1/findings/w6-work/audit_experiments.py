import os
import json

experiments_dir = "experiments"
exp_list = sorted(os.listdir(experiments_dir))

print(f"Total experiments directories: {len(exp_list)}")

validated_checks = []
unsubmitted_beats = []
promising_screens = []

# Public best scores currently:
public_bests = {
    "muse_bf16_proj": 1.5395,
    "muse_attention_decode": 2.6177,
    "muse_moe_w4a16": 2.3820,
    "muse_decode_layer": 2.2603,
    "muse_prefill_layer": 1.7874
}

for exp in exp_list:
    p = os.path.join(experiments_dir, exp)
    if not os.path.isdir(p):
        continue
    
    status_file = os.path.join(p, "status.json")
    manifest_file = os.path.join(p, "manifest.json")
    
    if os.path.exists(status_file):
        try:
            with open(status_file) as f:
                st = json.load(f)
            # check if it was a check or run
            # look for problem, mode, speedup, passed, tests, timed_cases
            # st might have 'result' or 'job'
            job = st.get("job", {})
            result = st.get("result", {})
            passed = result.get("passed", False)
            speedup = result.get("speedup", 0.0)
            mode = job.get("mode") or st.get("mode")
            problem = job.get("problem") or st.get("problem")
            job_id = job.get("job_id") or st.get("job_id") or st.get("id")
            
            # Let's collect check runs
            if mode == "check" and passed and speedup:
                validated_checks.append((exp, problem, speedup, job_id, result.get("speedup_ci")))
                if problem in public_bests and speedup > public_bests[problem]:
                    unsubmitted_beats.append((exp, problem, speedup, public_bests[problem], job_id, result.get("speedup_ci")))
        except Exception as e:
            pass

print(f"\nTotal validated checks found: {len(validated_checks)}")
print(f"Total unsubmitted checks beating published best: {len(unsubmitted_beats)}")
for item in unsubmitted_beats:
    print(f"  {item[0]}: problem={item[1]}, check_speedup={item[2]:.5f} vs pub_best={item[3]:.4f}, job={item[4]}, CI={item[5]}")
