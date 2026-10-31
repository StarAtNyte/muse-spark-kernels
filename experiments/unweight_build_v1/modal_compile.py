"""Compile a pinned local Unweight checkout on the public CUDA 13 image, without a GPU."""
import os,subprocess,modal
from pathlib import Path
checkout=os.environ.get('UNWEIGHT_CHECKOUT','/tmp/muse-unweight')
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
 "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
 .add_local_dir(checkout,'/unweight'))
app=modal.App('muse-unweight-cpu-build',image=image)
@app.function(timeout=480)
def build():
 p=subprocess.run(['timeout','360s','make','-C','/unweight','-j2'],capture_output=True,text=True)
 return f'RETURNCODE {p.returncode}\nSTDOUT\n{p.stdout[-12000:]}\nSTDERR\n{p.stderr[-12000:]}'
@app.local_entrypoint()
def main():
 if not Path(checkout,'src','unweight.cu').exists():raise RuntimeError('Set UNWEIGHT_CHECKOUT to a recursively cloned Unweight checkout')
 print(build.remote())
