"""Probe 3: Fix two bugs from probe 2.

Bug 1: ncu --import fails with 'Invalid option --clock-control specified for
--import' because the system's ncu wrapper adds --clock-control to every call.
Fix: find the real ncu binary (not the wrapper) and use it for all --import.

Bug 2: --set source doesn't collect gpu__time_duration.sum, so stall_tools
sees duration_ns=0 and reports mis-aim. Fix: use --set full.

Also uses --launch-count 8 (stall_price_run.py recipe) for pooling.
"""
import subprocess
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import torch
import sass_tools
import launch_tools
import sched_live_tools
import stall_tools
import orch_tools
import struct_tools
import role_exposure_tools

REP = "/tmp/p3_ncu"
KERN = "regex:attn_kernel.*8.*0.*15"

# ---- Discover the real ncu binary (bypass wrapper) ----
print("=== NCU BINARY INVESTIGATION ===")
ncu_path = shutil.which("ncu") or "ncu"
print(f"ncu in PATH: {ncu_path}")

ncu_real = ncu_path
# If the PATH entry is a script, find the ELF binary
try:
    with open(ncu_path, "rb") as f:
        header = f.read(4)
    is_script = header[:2] == b"#!"
    print(f"Is shell script: {is_script}")
    if is_script:
        with open(ncu_path) as f:
            script_text = f.read(2000)
        print("=== wrapper script ===")
        print(script_text[:2000])
        print("=== end wrapper ===")
except Exception as e:
    print(f"Could not inspect ncu binary: {e}")
    is_script = False

# Try known real binary paths
for candidate in [
    "/usr/local/cuda/bin/ncu",
    "/opt/nvidia/nsight-compute/ncu",
    "/usr/local/nsight-compute/ncu",
]:
    if os.path.exists(candidate):
        try:
            with open(candidate, "rb") as f:
                h = f.read(4)
            print(f"Found: {candidate}  is_script={h[:2] == b'#!'}")
            if h[:2] != b"#!" and h[:4] == b"\x7fELF":
                ncu_real = candidate
                print(f"Using real ELF binary: {ncu_real}")
                break
        except Exception as e:
            print(f"  cannot inspect {candidate}: {e}")

# Also check where nsight-compute puts ncu
find_r = subprocess.run(
    ["find", "/usr/local/cuda", "/opt", "/usr/local/nsight-compute",
     "-maxdepth", "6", "-name", "ncu"],
    capture_output=True, text=True, timeout=10,
)
print("find ncu:", find_r.stdout[:400] or "(none)")

print(f"Final ncu_real={ncu_real}")
# Quick test: does --import work without --clock-control?
test_r = subprocess.run(
    [ncu_real, "--version"],
    capture_output=True, text=True, timeout=10,
)
print("ncu --version:", test_r.stdout[:200])

# ---- Collection: --set full, launch-count 8 ----
print("\n=== Running NCU collection (--set full, --launch-count 8) ===")
print("KERN filter:", KERN)
ncu_cmd = [
    "ncu",
    "--set", "full",
    "--target-processes", "all",
    "--metrics",
    "launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,"
    "launch__cluster_max_active,launch__grid_dim_x,launch__grid_dim_y,launch__grid_dim_z,"
    "launch__block_dim_x,launch__block_dim_y,launch__block_dim_z,"
    "launch__registers_per_thread,launch__shared_mem_per_block_dynamic",
    "--kernel-name-base", "demangled",
    "--kernel-name", KERN,
    "--launch-count", "8",
    "-f",
    "-o", REP,
    "--profile-from-start", "off",
    "python3", "eval/ncu_run.py",
]
env = dict(os.environ, FK_LB_WORKLOAD_SCOPE="verify1")
r = subprocess.run(ncu_cmd, capture_output=True, text=True, env=env, timeout=300)
print("NCU stdout:", r.stdout[:3000] if r.stdout else "(empty)")
print("NCU stderr:", r.stderr[:2000] if r.stderr else "(empty)")
print("NCU return code:", r.returncode)

rep = REP + ".ncu-rep"
if not os.path.exists(rep):
    # retry without workload scope
    print("No report; retrying without FK_LB_WORKLOAD_SCOPE...")
    r2 = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
    print("Retry stdout:", r2.stdout[:2000])
    print("Retry stderr:", r2.stderr[:1000])

if not os.path.exists(rep):
    print(f"FATAL: no report at {rep}")
    sys.exit(1)

print(f"Report file: {rep}, size={os.path.getsize(rep)}")

# ---- §1: Launch metrics using real ncu binary ----
print("\n=== §1 LAUNCH METRICS ===")
raw_page = subprocess.run(
    [ncu_real, "--import", rep, "--page", "raw"],
    capture_output=True, text=True
)
print(f"§1 rc={raw_page.returncode}")
raw_text = raw_page.stdout
lines_raw = raw_text.splitlines()
print(f"Raw page lines: {len(lines_raw)}")
# Show first 60 lines
for line in lines_raw[:60]:
    print(line)
if len(lines_raw) > 60:
    print(f"... ({len(lines_raw) - 60} more lines)")
launch_ok = launch_tools.report(raw_text)
print(f"launch_ok={launch_ok}")

# ---- §2: SASS extraction using real ncu binary ----
print("\n=== §2 SASS ===")
sass_page = subprocess.run(
    [ncu_real, "--import", rep, "--page", "source", "--print-source", "sass"],
    capture_output=True, text=True, timeout=300
)
print(f"SASS rc={sass_page.returncode}")
sass_text = sass_page.stdout
lines = sass_text.splitlines()
print(f"SASS page total lines: {len(lines)}")
print("--- First 40 lines ---")
for line in lines[:40]:
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
    sass_text,
    profiled_kernel="void <unnamed>::attn_kernel<8, 0, 15>(<unnamed>::Params)",
    launch_pairs=launch_pairs,
)
print(f"sass_ok={sass_ok}")

# ---- §2d/2e: Schedule + liveness ----
print("\n=== §2d/2e SCHEDULE + LIVENESS ===")
sched_ok, live_ok = sched_live_tools.report(sass_text)
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
orch_ok = orch_tools.report(sass_text)
print(f"orch_ok={orch_ok}")

# ---- §2h: STRUCT table ----
print("\n=== §2h STRUCT TABLE ===")
csv_result = subprocess.run(
    [ncu_real, "--import", rep, "--page", "source", "--print-source", "sass", "--csv"],
    capture_output=True, text=True, timeout=300
)
print(f"CSV rc={csv_result.returncode}")
csv_text = csv_result.stdout
print(f"CSV text length: {len(csv_text)} chars")
sass_text2, counts = struct_tools.source_csv_to_sass_and_counts(csv_text)
struct_ok = struct_tools.report(
    sass_text2, counts,
    kernel="void <unnamed>::attn_kernel<8, 0, 15>(<unnamed>::Params)",
    ncta=33,
)
print(f"struct_ok={struct_ok}")
