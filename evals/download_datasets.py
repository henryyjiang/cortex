"""
Download eval datasets to local disk for use on compute nodes without internet.

Run once on a login node:
    HF_TOKEN=hf_... python evals/download_datasets.py

HF_TOKEN is required if the dataset is gated or rate-limited.
Get one at https://huggingface.co/settings/tokens
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent.parent / "data"
ROOT.mkdir(exist_ok=True)

token = os.environ.get("HF_TOKEN") or None

print("Downloading xiaowu0162/LongMemEval ...")
local_dir = snapshot_download(
    repo_id   = "xiaowu0162/LongMemEval",
    repo_type = "dataset",
    local_dir = str(ROOT / "LongMemEval"),
    token     = token,
)
print(f"Saved → {local_dir}")
