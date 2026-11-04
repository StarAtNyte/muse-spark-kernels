"""Single-invocation Modal runner (judge image, one H200). Usage (from the repo root):
modal run experiments/moe_small_v3/run_modal.py --sub experiments/moe_small_v3/submission.py --drv experiments/moe_small_v3/driver.py"""
import os
import modal
ROOT = os.getcwd()
K = f"{ROOT}/kernels"
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
for f in ("reference.py", "checks.py", "nvfp4.py"):
    image = image.add_local_file(f"{K}/muse_moe_w4a16/{f}", f"/w/kernels/muse_moe_w4a16/{f}")
app = modal.App("moe-small-v3", image=image)

@app.function(gpu="H200", timeout=420, cpu=16)
def run(sub: str, driver: str, env: str):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    for n, s in (("sub.py", sub), ("drv.py", driver)):
        open(f"/w/run/{n}", "w").write(s)
    e = dict(os.environ); e["DRV_ARGS"] = env
    p = subprocess.run(["timeout", "400s", "python", "/w/run/drv.py"], cwd="/w/run", capture_output=True, text=True, env=e)
    return f"RETURNCODE {p.returncode}\nSTDOUT:\n{p.stdout[-300000:]}\nSTDERR:\n{p.stderr[-12000:]}"

@app.local_entrypoint()
def main(sub: str, drv: str, args: str = ""):
    print(run.remote(open(sub).read(), open(drv).read(), args))
