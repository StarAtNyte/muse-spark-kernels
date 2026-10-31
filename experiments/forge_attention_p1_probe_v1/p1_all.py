"""Probe 1: All-in-one SASS + launch + schedule + stall for attn_kernel<8,0,15>.

Targets verify1 (batch=1, length=8) which launches the dominant kernel.
Collects --set source plus explicit cluster metrics.
"""
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import sass_tools
import launch_tools
import sched_live_tools
import stall_tools
import orch_tools
import struct_tools
import role_exposure_tools

REP = "/tmp/p1_all"
KERN = "regex:attn_kernel.*8.*0.*15"

# ---- collect: --set source + cluster metrics ----
ncu_cmd = [
    "ncu",
    "--set", "source",
    "--metrics",
    "launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,"
    "launch__cluster_max_active,launch__grid_dim_x,launch__grid_dim_y,launch__grid_dim_z,"
    "launch__block_dim_x,launch__block_dim_y,launch__block_dim_z,"
    "launch__registers_per_thread,launch__shared_mem_per_block_dynamic",
    "--kernel-name-base", "function",
    "--kernel-name", KERN,
    "--launch-count", "1",
    "-f",
    "-o", REP,
    "--profile-from-start", "off",
    "python3", "eval/ncu_run.py",
]
print("=== Running NCU collection ===")
env = dict(os.environ, FK_LB_WORKLOAD_SCOPE="verify1")
r = subprocess.run(ncu_cmd, capture_output=True, text=True, env=env)
print("NCU stdout:", r.stdout[:3000] if r.stdout else "(empty)")
print("NCU stderr:", r.stderr[:2000] if r.stderr else "(empty)")
print("NCU return code:", r.returncode)

rep = REP + ".ncu-rep"
if not os.path.exists(rep):
    # retry without workload scope (will run the default representative)
    print("No report with verify1 scope, retrying without scope restriction...")
    r2 = subprocess.run(ncu_cmd[:-2] + ["python3", "eval/ncu_run.py"],
                        capture_output=True, text=True)
    print("Retry stdout:", r2.stdout[:2000])
    print("Retry stderr:", r2.stderr[:1000])

if not os.path.exists(rep):
    print(f"FATAL: no report at {rep}")
    sys.exit(1)

print(f"Report file: {rep}, size={os.path.getsize(rep)}")

# ---- §1: Launch metrics ----
print("\n=== §1 LAUNCH METRICS ===")
raw_page = subprocess.run(
    ["ncu", "--import", rep, "--page", "raw"],
    capture_output=True, text=True
).stdout
print(raw_page[:5000])
launch_ok = launch_tools.report(raw_page)
print(f"launch_ok={launch_ok}")

# ---- §2: SASS page (first 30 lines raw, then analysis) ----
print("\n=== §2 SASS ===")
sass_page = subprocess.run(
    ["ncu", "--import", rep, "--page", "source", "--print-source", "sass"],
    capture_output=True, text=True
).stdout

lines = sass_page.splitlines()
print(f"SASS page total lines: {len(lines)}")
print("--- First 30 lines ---")
for line in lines[:30]:
    print(line)
print("--- Last 20 lines ---")
for line in lines[-20:]:
    print(line)

launch_pairs = [
    ("void <unnamed>::attn_kernel<8, 0, 15>(<unnamed>::Params)", 3900.0),
    ("void <unnamed>::attn_kernel<1, 0, 1>(<unnamed>::Params)", 3800.0),
    ("void <unnamed>::attn_kernel<1, 0, 13>(<unnamed>::Params)", 2900.0),
]
sass_ok = sass_tools.analyze(
    sass_page,
    profiled_kernel="void <unnamed>::attn_kernel<8, 0, 15>(<unnamed>::Params)",
    launch_pairs=launch_pairs,
)
print(f"sass_ok={sass_ok}")

# ---- §2d/2e: Schedule + liveness ----
print("\n=== §2d/2e SCHEDULE + LIVENESS ===")
sched_ok, live_ok = sched_live_tools.report(sass_page)
print(f"sched_ok={sched_ok}, live_ok={live_ok}")

# ---- §2f: Stall ledger ----
print("\n=== §2f STALL LEDGER ===")
stall_ok = stall_tools.price_report(rep)
print(f"stall_ok={stall_ok}")
print("\n--- Segment ledger ---")
stall_tools.segment_ledger(rep)
print("\n--- Role exposure (PARTIAL without PM) ---")
role_exposure_tools.report(rep)

# ---- §2g: Orch dossier ----
print("\n=== §2g ORCH DOSSIER ===")
orch_ok = orch_tools.report(sass_page)
print(f"orch_ok={orch_ok}")

# ---- §2h: STRUCT table ----
print("\n=== §2h STRUCT TABLE ===")
csv_text = subprocess.run(
    ["ncu", "--import", rep, "--page", "source", "--print-source", "sass", "--csv"],
    capture_output=True, text=True
).stdout
print(f"CSV text length: {len(csv_text)} chars")
sass_text2, counts = struct_tools.source_csv_to_sass_and_counts(csv_text)
struct_ok = struct_tools.report(
    sass_text2, counts,
    kernel="void <unnamed>::attn_kernel<8, 0, 15>(<unnamed>::Params)",
    ncta=33,
)
print(f"struct_ok={struct_ok}")
