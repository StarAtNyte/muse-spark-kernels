"""Run pinned Humming research source on the public challenge image."""
import os
from pathlib import Path
import subprocess
import modal

ROOT = Path.cwd()
HUMMING = Path('/tmp/muse-humming-research')
expected = '948e6fd547e703065be51529ac1db5569256774e'
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
).add_local_dir(str(HUMMING / 'humming'), '/w/humming/humming')
 .add_local_dir(str(ROOT / 'kernels' / 'muse_prefill_layer'), '/w/kernels/muse_prefill_layer'))
app = modal.App('muse-humming-stage-probe', image=image)
@app.function(gpu='H200', timeout=360)
def run(driver: str):
    import subprocess
    Path('/w/run').mkdir(exist_ok=True)
    Path('/w/run/driver.py').write_text(driver)
    p=subprocess.run(['timeout','300s','python','/w/run/driver.py'],capture_output=True,text=True)
    return f'RETURNCODE {p.returncode}\n{p.stdout[-200000:]}\nSTDERR:\n{p.stderr[-10000:]}'
@app.local_entrypoint()
def main():
    actual = subprocess.check_output(['git', '-C', str(HUMMING), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise RuntimeError(f'Humming checkout mismatch: {actual}')
    print(run.remote((ROOT / 'experiments/humming_prefill_stage_profile_v1/driver.py').read_text()))
