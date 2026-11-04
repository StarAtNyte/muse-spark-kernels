"""Single-invocation Modal runner (judge image, one H200, 300 s function timeout)."""
import os
import modal
ROOT = os.getcwd()
K = f"{ROOT}/kernels"
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
for f in ("reference.py", "checks.py", "nvfp4.py"):
    image = image.add_local_file(f"{K}/muse_moe_w4a16/{f}", f"/w/kernels/muse_moe_w4a16/{f}")
app = modal.App("claude-moe-fp32-opt-v1", image=image)

@app.function(gpu="H200", timeout=300, cpu=16)
def run(parent: str, cand: str, driver: str):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    for n, s in (("parent.py", parent), ("cand.py", cand), ("drv.py", driver)):
        open(f"/w/run/{n}", "w").write(s)
    p = subprocess.run(["timeout", "280s", "python", "/w/run/drv.py"], cwd="/w/run", capture_output=True, text=True)
    return f"RETURNCODE {p.returncode}\nSTDOUT:\n{p.stdout[-300000:]}\nSTDERR:\n{p.stderr[-8000:]}"

@app.local_entrypoint()
def main():
    d = f"{ROOT}/experiments/claude_moe_fp32_opt_v1"
    print(run.remote(open(f"{ROOT}/experiments/scored_moe_fp32_ybuf_v1/submission.py").read(), open(f"{d}/submission.py").read(), open(f"{d}/driver.py").read()))
