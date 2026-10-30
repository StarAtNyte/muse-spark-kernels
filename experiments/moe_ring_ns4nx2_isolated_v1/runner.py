"""Like run_cuda13_h200.py, but runs the driver once per environment set (each in its own process and container)."""
import os, json
import modal
ROOT = os.getcwd()
image = (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu22.04', add_python='3.12')
         .apt_install('ninja-build', 'libnuma1')
         .pip_install('torch==2.11.0', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy', 'triton')
         .add_local_dir(f'{ROOT}/kernels', '/w/kernels')
         .add_local_dir(f'{ROOT}/dev/modal', '/w/modal'))
app = modal.App('muse-cuda13-env', image=image)

@app.function(gpu='H200', timeout=360)
def run(src: str, driver: str, env: dict):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    open('/w/run/sub.py', 'w').write(src); open('/w/run/drv.py', 'w').write(driver)
    p = subprocess.run(['timeout', '300s', 'python', '/w/run/drv.py'], cwd='/w/run', capture_output=True, text=True,
                       env={**os.environ, **env})
    return f'ENV {json.dumps(env)} RETURNCODE {p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-3000:]}'

@app.local_entrypoint()
def main(file: str, drv: str, envs: str):
    src, d = open(file).read(), open(drv).read()
    for out in run.starmap([(src, d, e) for e in json.loads(envs)]):
        print(out)
