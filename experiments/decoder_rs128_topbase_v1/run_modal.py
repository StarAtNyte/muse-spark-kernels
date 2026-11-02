"""Run a paired H200 screen of rs128 on the exact 2.26032x scored decoder."""
import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(os.environ.get("MUSE_SPARK_ROOT", "/home/nitiz/Projects/Self/Challenges/muse-spark"))
EXP = ROOT / "experiments" / "decoder_rs128_topbase_v1"
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    .run_commands(
        "python -c 'import sysconfig; from pathlib import Path; "
        "Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text("
        "\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
    )
    .add_local_dir(str(ROOT / "kernels"), "/w/kernels")
    .add_local_dir(str(EXP), "/w/exp")
)
app = modal.App("muse-decoder-rs128-topbase-screen", image=image)


@app.function(gpu="H200", timeout=360)
def run(cases_json: str):
    env = {**os.environ, "CASES": cases_json}
    p = subprocess.run(
        ["timeout", "330s", "python", "/w/exp/driver.py"],
        cwd="/w/exp",
        capture_output=True,
        text=True,
        env=env,
    )
    return f"RETURNCODE {p.returncode}\n{p.stdout[-100000:]}\nSTDERR:\n{p.stderr[-5000:]}"


@app.local_entrypoint()
def main():
    cases = [
        {"batch": 16, "q_len": 8, "layer": "sliding"},
        {"batch": 16, "q_len": 8, "layer": "full"},
        {"batch": 32, "q_len": 8, "layer": "sliding"},
        {"batch": 32, "q_len": 8, "layer": "full"},
    ]
    print(run.remote(json.dumps(cases)))
