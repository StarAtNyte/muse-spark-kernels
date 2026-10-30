"""Inspect the official public judge image on H200 without installing packages.

The service's recorded image digest and exact benchmark harness remain authoritative.
"""
import modal

image = modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').run_commands(
    "python -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text(\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
)
app = modal.App('muse-judge-image-audit', image=image)

@app.function(gpu='H200', timeout=120)
def inspect_environment():
    import importlib.metadata as metadata
    import json
    import platform
    import subprocess
    import sys
    import torch

    def command(*args):
        try:
            p = subprocess.run(args, text=True, capture_output=True, timeout=15)
            return {'rc': p.returncode, 'stdout': p.stdout[-4000:], 'stderr': p.stderr[-1000:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            return {'error': str(e)}

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {'python': sys.version, 'platform': platform.platform(),
            'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0),
            'capability': torch.cuda.get_device_capability(0),
            'packages': {name: version(name) for name in ('triton', 'sglang', 'flashinfer-python', 'flash-attn', 'nvidia-cutlass-dsl', 'tilelang', 'deep-gemm', 'numpy')},
            'nvidia_smi': command('nvidia-smi', '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.memory,memory.total', '--format=csv'),
            'nvcc': command('nvcc', '--version')}

@app.local_entrypoint()
def main():
    import json
    print(json.dumps(inspect_environment.remote(), indent=2, sort_keys=True))
