"""
Download eval datasets to local disk for use on compute nodes without internet.

Run once on a login node:
    python evals/download_datasets.py
"""
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).parent.parent / "data"
ROOT.mkdir(exist_ok=True)

print("Downloading xiaowu0162/LongMemEval ...")
ds = load_dataset("xiaowu0162/LongMemEval")
ds.save_to_disk(ROOT / "LongMemEval")
print(f"Saved → {ROOT / 'LongMemEval'}")
