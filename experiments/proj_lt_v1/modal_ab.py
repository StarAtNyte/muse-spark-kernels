"""Paired A/B on one H200: subprocesses A, B, A, B in one container (separate extension dirs)."""
import json, os
import modal
ROOT = os.getcwd()
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
).add_local_dir(f'{ROOT}/kernels', '/w/kernels'))
app = modal.App('muse-proj-ab', image=image)

@app.function(gpu='H200', timeout=1800)
def run(srcs: dict, driver: str, order: list):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    for k, s in srcs.items():
        open(f'/w/run/{k}.py', 'w').write(s)
    open('/w/run/drv.py', 'w').write(driver)
    log = ''
    for i, k in enumerate(order):
        env = {**os.environ, 'SUB': f'/w/run/{k}.py', 'TAG': f'{k}{i}', 'TORCH_EXTENSIONS_DIR': f'/tmp/ext_{k}'}
        p = subprocess.run(['timeout', '600s', 'python', '/w/run/drv.py'], cwd='/w/run', capture_output=True, text=True, env=env)
        log += f'=== {k}{i} rc={p.returncode}\n{p.stdout[-250000:]}\nSTDERR:\n{p.stderr[-3000:]}\n'
    return log

@app.local_entrypoint()
def main(a: str, b: str, drv: str, order: str = 'A,B,A,B'):
    srcs = {'A': open(a).read(), 'B': open(b).read()}
    print(run.remote(srcs, open(drv).read(), order.split(',')))
