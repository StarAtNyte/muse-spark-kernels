"""Bounded CPU-only compilation probe for the experimental 128-row WGS kernel."""
import os
import modal

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("ninja-build", "libnuma1")
         .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy", "triton"))
app = modal.App("muse-moe-cpu-compile", image=image)


@app.function(cpu=8, memory=32768, timeout=600)
def probe(src: str):
    import subprocess
    os.makedirs("/w/run", exist_ok=True)
    open("/w/run/sub.py", "w").write(src)
    driver = '''import importlib.util, torch
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
spec = importlib.util.spec_from_file_location("candidate", "/w/run/sub.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("COMPILE_START", flush=True)
m._ext()
print("COMPILE_DONE", flush=True)
'''
    open("/w/run/driver.py", "w").write(driver)
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="9.0", MAX_JOBS="4")
    p = subprocess.run(["timeout", "450s", "python", "/w/run/driver.py"],
                       capture_output=True, text=True, env=env)
    return {"returncode": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-12000:]}


@app.local_entrypoint()
def main(file: str):
    import json
    print(json.dumps(probe.remote(open(file).read())))
