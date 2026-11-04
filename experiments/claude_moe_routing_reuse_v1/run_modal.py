"""Single-use Modal runner: judge image, one H200, hard 240 s function timeout. Mounts only the reference files."""
import os
import modal

ROOT = os.getcwd()
K = f"{ROOT}/kernels"
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
for prob in ("muse_decode_layer", "muse_moe_w4a16"):
    for f in ("reference.py", "checks.py", "nvfp4.py"):
        image = image.add_local_file(f"{K}/{prob}/{f}", f"/w/kernels/{prob}/{f}")
app = modal.App("claude-moe-routing-reuse-v1", image=image)


@app.function(gpu="H200", timeout=240)
def run(src: str):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    open("/w/run/probe.py", "w").write(src)
    p = subprocess.run(["timeout", "225s", "python", "/w/run/probe.py", "170"], cwd="/w/run", capture_output=True, text=True)
    return f"RETURNCODE {p.returncode}\n{p.stdout[-400000:]}\nSTDERR:\n{p.stderr[-6000:]}"


@app.local_entrypoint()
def main(file: str):
    print(run.remote(open(file).read()))
