"""Run a bounded diagnostic on H200 using the challenge's public SGLang image.

Component drivers are screens, not replacements for the official judge harness.
"""
import json
import os
import modal

ROOT = os.getcwd()
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
         .add_local_dir(f'{ROOT}/kernels', '/w/kernels')
         .add_local_dir(f'{ROOT}/dev/modal', '/w/modal'))
app = modal.App('muse-judge-image-diagnostic', image=image)

@app.function(gpu='H200', timeout=360)
def run(src: str, driver: str, env: dict):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    with open('/w/run/sub.py', 'w') as f:
        f.write(src)
    with open('/w/run/drv.py', 'w') as f:
        f.write(driver)
    p = subprocess.run(['timeout', '300s', 'python', '/w/run/drv.py'], cwd='/w/run',
                       capture_output=True, text=True, env={**os.environ, **env})
    return f'ENV {json.dumps(env)} RETURNCODE {p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-5000:]}'

@app.local_entrypoint()
def main(file: str, drv: str, envs: str = '[{}]'):
    with open(file) as f:
        src = f.read()
    with open(drv) as f:
        driver = f.read()
    for result in run.starmap([(src, driver, env) for env in json.loads(envs)]):
        print(result)
