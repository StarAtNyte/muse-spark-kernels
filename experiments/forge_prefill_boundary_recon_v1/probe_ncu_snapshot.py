"""NCU --set source collection and full analysis of boundary_kernel.

Runs NCU against eval/ncu_run.py (--profile-from-start off, --launch-count 1),
extracts the SASS page, then calls all seeded analysis tools for the wrap-up
verdict lines: LAUNCH-OK/PARTIAL, SASS-OK/FAIL, ORCH-OK, SCHED-OK, LIVE-OK,
STALL-OK, ROLE-OK/PARTIAL, STRUCT-OK/WAIVED.
"""
import os, sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENTRY = str(ROOT / "eval" / "ncu_run.py")

REP = "/tmp/bnd_rep"
REP_FILE = REP + ".ncu-rep"
KERNEL_RE = "regex:boundary_kernel"
NCU_METRICS_EXTRA = (
    "launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,"
    "launch__cluster_max_active,"
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed,"
    "dram__bytes_read.sum,dram__bytes_write.sum,"
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,"
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
)

# -------------------------------------------------------------------------
print("=== Step 1: NCU --set source collection ===")
cmd = [
    "ncu",
    "--set", "source",
    "--metrics", NCU_METRICS_EXTRA,
    "--clock-control", "none",
    "--kernel-name-base", "function",
    "--kernel-name", KERNEL_RE,
    "--launch-count", "1",
    "--profile-from-start", "off",
    "--target-processes", "all",
    "-f", "-o", REP,
    sys.executable, ENTRY,
]
r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=240)
print("returncode:", r.returncode)
stdout_tail = r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout
print("STDOUT:\n", stdout_tail)
if r.stderr:
    print("STDERR (last 1500):\n", r.stderr[-1500:])

if not Path(REP_FILE).exists():
    print(f"FATAL: {REP_FILE} not created; aborting")
    sys.exit(1)

# -------------------------------------------------------------------------
print("\n=== Step 2: SASS page extraction ===")
sass_cmd = [
    "ncu", "--import", REP_FILE,
    "--page", "source",
    "--print-source", "sass",
]
rs = subprocess.run(sass_cmd, capture_output=True, text=True, timeout=120)
sass_text = rs.stdout
print(f"SASS: {len(sass_text)} chars, {len(sass_text.splitlines())} lines")
if rs.returncode != 0:
    print("ncu import STDERR:", rs.stderr[:600])
print("=== SASS first 40 lines ===")
for i, ln in enumerate(sass_text.splitlines()[:40]):
    print(f"  {i:3d}: {ln}")

# -------------------------------------------------------------------------
print("\n=== Step 3: launch_tools (cluster / grid / block) ===")
sys.path.insert(0, str(HERE))
import launch_tools
launch_tools.report(sass_text, kernel="boundary_kernel")

# -------------------------------------------------------------------------
print("\n=== Step 4: sass_tools (opcode histogram + self-checks) ===")
import sass_tools
# boundary_kernel has no MMA families; pass families=() to skip that check.
# Device median is 27200 ns — used as the single launch-pair duration.
sass_tools.analyze(sass_text,
                   profiled_kernel="boundary_kernel",
                   launch_pairs=[("boundary_kernel", 27200)],
                   families=(),
                   min_instructions=100)

# -------------------------------------------------------------------------
print("\n=== Step 5: orch_tools (sync topology, supply, cache policy) ===")
import orch_tools
orch_tools.report(sass_text)

# -------------------------------------------------------------------------
print("\n=== Step 6: sched_live_tools (schedule order + live-range) ===")
import sched_live_tools
sched_live_tools.report(sass_text)

# -------------------------------------------------------------------------
print("\n=== Step 7: stall_tools price_report ===")
import stall_tools
stall_tools.price_report(REP_FILE)

print("\n=== Step 7b: stall_tools segment_ledger ===")
stall_tools.segment_ledger(REP_FILE)

# -------------------------------------------------------------------------
print("\n=== Step 8: role_exposure_tools ===")
import role_exposure_tools
role_exposure_tools.report(REP_FILE)

# -------------------------------------------------------------------------
print("\n=== Step 9: struct_tools (CSV source page) ===")
import struct_tools
csv_cmd = [
    "ncu", "--import", REP_FILE,
    "--page", "source",
    "--print-source", "sass",
    "--csv",
]
rc = subprocess.run(csv_cmd, capture_output=True, text=True, timeout=120)
csv_text = rc.stdout
print(f"CSV: {len(csv_text)} chars")
if rc.returncode != 0:
    print("ncu csv STDERR:", rc.stderr[:400])
sass2, counts = struct_tools.source_csv_to_sass_and_counts(csv_text)
struct_tools.report(sass2, counts, kernel="boundary_kernel", ncta=753)

# -------------------------------------------------------------------------
print("\n=== Step 10: extra DRAM + cluster metrics ===")
metrics_cmd = [
    "ncu", "--import", REP_FILE,
    "--metrics", NCU_METRICS_EXTRA,
    "--csv",
]
rm = subprocess.run(metrics_cmd, capture_output=True, text=True, timeout=60)
print("extra metrics CSV:\n", rm.stdout[:3000])
if rm.returncode != 0:
    print("metrics STDERR:", rm.stderr[:400])

print("\n=== DONE ===")
