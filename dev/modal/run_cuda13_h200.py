"""Run a local source and diagnostic driver on an H200 with CUDA 13."""
import os
import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
image = (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu22.04', add_python='3.12')
         .apt_install('ninja-build', 'libnuma1')
         .pip_install('torch==2.11.0', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy', 'triton')
         .add_local_dir(f'{ROOT}/kernels', '/w/kernels')
         .add_local_dir(f'{ROOT}/dev/modal', '/w/modal'))
app = modal.App('muse-cuda13-h200', image=image)

@app.function(gpu='H200', timeout=900)
def run(src: str, driver: str):
    import os, subprocess
    os.makedirs('/w/run', exist_ok=True)
    open('/w/run/sub.py', 'w').write(src)
    open('/w/run/drv.py', 'w').write(driver)
    p = subprocess.run(['timeout', '600s', 'python', '/w/run/drv.py'], cwd='/w/run',
                       capture_output=True, text=True)
    return f'RETURNCODE {p.returncode}\n{p.stdout[-900000:]}\nSTDERR:\n{p.stderr[-16000:]}'

@app.local_entrypoint()
def main(file: str, drv: str):
    print(run.remote(open(file).read(), open(drv).read()))
