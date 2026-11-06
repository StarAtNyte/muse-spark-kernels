"""Serving-track prep on Modal: cache weights and JIT builds so later 8xH200 runs start fast.

Stages (run in order, each is resumable):
    modal run dev/modal/serving_prep.py::check_access   # CPU only: can the HF secret read the gated model?
    modal run dev/modal/serving_prep.py::download       # CPU only: model + DFlash drafter into a Volume

The Volume holds the weights so no later GPU session pays for the 540 GB download.
"""

import modal

MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
DRAFTER = "RadixArk/Muse-Spark-Open-DFlash2"
WEIGHTS_DIR = "/weights"

app = modal.App("muse-serving-prep")
weights = modal.Volume.from_name("muse-serving-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]==0.26.2"
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})


@app.function(image=cpu_image, secrets=[hf_secret], timeout=300)
def check_access():
    from huggingface_hub import HfApi, hf_hub_download

    for repo in (MODEL, DRAFTER):
        path = hf_hub_download(repo, "config.json")
        size = sum(f.size or 0 for f in HfApi().model_info(repo, files_metadata=True).siblings)
        print(f"{repo}: readable ({path}), {size / 1e9:.1f} GB")


@app.function(image=cpu_image, secrets=[hf_secret], volumes={WEIGHTS_DIR: weights},
              timeout=4 * 3600, cpu=8, memory=16384)
def download():
    from huggingface_hub import snapshot_download

    for repo in (DRAFTER, MODEL):
        local = f"{WEIGHTS_DIR}/{repo}"
        snapshot_download(repo, local_dir=local, max_workers=16)
        weights.commit()  # persist each repo as soon as it lands, so a timeout loses nothing
        print(f"{repo}: done -> {local}")
