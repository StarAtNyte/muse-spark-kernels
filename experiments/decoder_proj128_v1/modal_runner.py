"""Temporary single-use runner for experiments/decoder_proj128_v1 (judge image, H200, 300 s function timeout)."""
import json
import os
import modal

ROOT = os.getcwd()
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
         .add_local_dir(f'{ROOT}/kernels', '/w/kernels'))
app = modal.App('claude-decoder-proj128-v1', image=image)

@app.function(gpu='H200', timeout=1200)
def run(src: str, driver: str):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    with open('/w/run/sub.py', 'w') as f:
        f.write(src)
    with open('/w/run/drv.py', 'w') as f:
        f.write(driver)
    p = subprocess.run(['timeout', '1150s', 'python', '/w/run/drv.py'], cwd='/w/run', capture_output=True, text=True)
    return f'RETURNCODE {p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-5000:]}'

@app.local_entrypoint()
def main(file: str, drv: str):
    print(run.remote(open(file).read(), open(drv).read()))
