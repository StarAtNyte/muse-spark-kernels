"""Build upstream codec in a CPU image step, then use one bounded H200 for correctness."""
import os, modal
from pathlib import Path
root=os.getcwd();checkout=os.environ.get('UNWEIGHT_CHECKOUT','/tmp/muse-unweight')
image=(modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands("python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'")
 .add_local_dir(checkout,'/unweight',copy=True)
 .run_commands('cd /unweight && /usr/local/cuda/bin/nvcc -shared -std=c++20 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --extended-lambda -Xcompiler -fPIC -DKITTENS_HOPPER -I src -I third-party/ThunderKittens/include -I third-party/ThunderKittens/prototype src/unweight.cu -L/usr/local/cuda/lib64/stubs -lcuda -o /opt/libunweight.so')
 .add_local_dir(f'{root}/kernels','/w/kernels')
 .add_local_file(f'{root}/experiments/unweight_roundtrip_v1/driver.py','/w/driver.py'))
app=modal.App('muse-unweight-roundtrip',image=image)
@app.function(gpu='H200',timeout=180)
def run():
 import subprocess
 p=subprocess.run(['timeout','120s','python','/w/driver.py'],capture_output=True,text=True)
 return f'RETURNCODE {p.returncode}\n{p.stdout}\nSTDERR\n{p.stderr[-12000:]}'
@app.local_entrypoint()
def main():
 if not Path(checkout,'src','unweight.cu').exists():raise RuntimeError('Clone Unweight recursively and set UNWEIGHT_CHECKOUT')
 print(run.remote())
