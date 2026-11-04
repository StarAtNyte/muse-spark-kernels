"""decoder_bf16_v1 runner: judge image, H200. Uploads the submission, a driver and helper modules; build caches on a Volume."""
import os
import modal

ROOT = os.getcwd()
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
         .add_local_dir(f'{ROOT}/kernels', '/w/kernels'))
app = modal.App('decoder-bf16-v1', image=image)
vol = modal.Volume.from_name('decoder-small-v2-cache', create_if_missing=True)


@app.function(gpu='H200', timeout=2000, volumes={'/cache': vol})
def run(files: dict, driver: str):
    import subprocess
    os.makedirs('/w/run', exist_ok=True)
    for name, src in files.items():
        with open(f'/w/run/{name}', 'w') as f:
            f.write(src)
    env = {**os.environ, 'TORCH_EXTENSIONS_DIR': '/cache/torch_ext', 'TRITON_CACHE_DIR': '/cache/triton'}
    p = subprocess.run(['timeout', '1900s', 'python', f'/w/run/{driver}'], cwd='/w/run', capture_output=True, text=True, env=env)
    vol.commit()
    return f'RETURNCODE {p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-6000:]}'


@app.local_entrypoint()
def main(sub: str, par: str, driver: str, extra: str = ''):
    d = os.path.dirname(os.path.abspath(driver))
    files = {'sub.py': open(sub).read(), 'par.py': open(par).read(), **{os.path.basename(x): open(x).read() for x in os.environ.get('VARS', '').split(',') if x}, os.path.basename(driver): open(driver).read()}
    for e in [x for x in extra.split(',') if x]:
        files[e] = open(os.path.join(d, e)).read()
    print(run.remote(files, os.path.basename(driver)))
