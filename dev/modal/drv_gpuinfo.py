import subprocess
print(subprocess.run("nvidia-smi --query-gpu=name,power.limit,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,memory.total,driver_version,pstate,clocks_event_reasons.active --format=csv", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("nvidia-smi -q | grep -i -E \"product name|power limit|ecc mode|mig mode|board part|persistence\" | head -20", shell=True, capture_output=True, text=True).stdout)
