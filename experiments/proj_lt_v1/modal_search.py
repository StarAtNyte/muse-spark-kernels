"""Parallel cuBLASLt search on H200 in the judge image: one container per env (CELL=layer:m)."""
import json, os
import modal
ROOT = os.getcwd()
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
).add_local_dir(f'{ROOT}/kernels', '/w/kernels'))
app = modal.App('muse-proj-ltsearch', image=image)

@app.function(gpu='H200', timeout=900)
def run(driver: str, env: dict):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    open('/w/run/drv.py', 'w').write(driver)
    p = subprocess.run(['timeout', '840s', 'python', '/w/run/drv.py'], cwd='/w/run', capture_output=True, text=True,
                       env={**os.environ, **env})
    return f'=== ENV {json.dumps(env)} rc={p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-4000:]}'

@app.local_entrypoint()
def main(drv: str, envs: str):
    d = open(drv).read()
    for r in run.starmap([(d, e) for e in json.loads(envs)]):
        print(r, flush=True)
