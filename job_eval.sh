#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex

RESULTS_DIR="eval_results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CORTEX_CKPT="runs/cortex-5b/checkpoint_0154441/checkpoint.pt"
CORTEX_K4_CKPT="runs/cortex-5b-k4/checkpoint_0152584/checkpoint.pt"
PYTHIA_CKPT="runs/pythia-5b/checkpoint_0152584/checkpoint.pt"
PARCAE_CKPT="runs/parcae-5b/checkpoint_0154441/checkpoint.pt"

# ── BABILong ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "BABILong evaluation"
echo "============================================================"

echo "[1/4] cortex-5b"
python eval_babilong.py \
    --checkpoint "$CORTEX_CKPT" \
    --memory_slots 0 \
    --T 8 \
    --out_dir "$RESULTS_DIR/babilong/cortex-5b"

echo "[2/4] cortex-5b-k4"
python eval_babilong.py \
    --checkpoint "$CORTEX_K4_CKPT" \
    --memory_slots 4 \
    --T 8 \
    --out_dir "$RESULTS_DIR/babilong/cortex-5b-k4"

echo "[3/4] pythia-5b"
python eval_babilong.py \
    --checkpoint "$PYTHIA_CKPT" \
    --memory_slots 0 \
    --T 1 \
    --out_dir "$RESULTS_DIR/babilong/pythia-5b"

echo "[4/4] parcae-5b"
python eval_babilong.py \
    --checkpoint "$PARCAE_CKPT" \
    --memory_slots 0 \
    --T 8 \
    --out_dir "$RESULTS_DIR/babilong/parcae-5b"

# ── LongMemEval ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "LongMemEval evaluation"
echo "============================================================"

echo "[1/4] cortex-5b"
python eval_longmemeval.py \
    --checkpoint "$CORTEX_CKPT" \
    --memory_slots 0 \
    --T 8 \
    --out_dir "$RESULTS_DIR/longmemeval/cortex-5b"

echo "[2/4] cortex-5b-k4"
python eval_longmemeval.py \
    --checkpoint "$CORTEX_K4_CKPT" \
    --memory_slots 4 \
    --T 8 \
    --out_dir "$RESULTS_DIR/longmemeval/cortex-5b-k4"

echo "[3/4] pythia-5b"
python eval_longmemeval.py \
    --checkpoint "$PYTHIA_CKPT" \
    --memory_slots 0 \
    --T 1 \
    --out_dir "$RESULTS_DIR/longmemeval/pythia-5b"

echo "[4/4] parcae-5b"
python eval_longmemeval.py \
    --checkpoint "$PARCAE_CKPT" \
    --memory_slots 0 \
    --T 8 \
    --out_dir "$RESULTS_DIR/longmemeval/parcae-5b"

# ── Aggregate results into tables ─────────────────────────────────────────────
echo "============================================================"
echo "Aggregating results"
echo "============================================================"

python - <<'PYEOF'
import json, os, sys
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "eval_results"))

MODELS = ["cortex-5b", "cortex-5b-k4", "pythia-5b", "parcae-5b"]
BENCHMARKS = {
    "babilong":    {"tasks": ["qa1", "qa2", "qa3"]},
    "longmemeval": {"tasks": [None]},   # single task, results at top level
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def print_table(title, rows, col_headers):
    """Print a plain-text table."""
    col_w = max(len(h) for h in col_headers + [r[0] for r in rows]) + 2
    header = f"{'Model':<20}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    sep    = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        label = row[0]
        vals  = row[1:]
        print(f"{label:<20}" + "".join(f"{v:>{col_w}}" for v in vals))
    print(sep)

# ── BABILong tables (one per task) ───────────────────────────────────────────
for task in ["qa1", "qa2", "qa3"]:
    # Collect all buckets from first available result
    buckets = None
    model_rows = []
    for model in MODELS:
        path = results_dir / "babilong" / model / "results.json"
        if not path.exists():
            print(f"  WARNING: missing {path}", file=sys.stderr)
            continue
        data = load_json(path)
        # results.json structure: {label: {task: {bucket: {correct, total, accuracy}}}}
        # We only ran one model per invocation so there's one key
        label_key = next(iter(data))
        task_data = data[label_key].get(task, {})
        if buckets is None:
            buckets = [b for b, r in task_data.items() if r["total"] > 0]
        row = [model] + [
            f"{task_data.get(b, {}).get('accuracy', 0):.3f}"
            for b in buckets
        ]
        model_rows.append(row)
    if buckets and model_rows:
        print_table(f"BABILong {task.upper()} — accuracy by context length", model_rows, buckets)

# ── LongMemEval table ────────────────────────────────────────────────────────
buckets = None
model_rows = []
for model in MODELS:
    path = results_dir / "longmemeval" / model / "results.json"
    if not path.exists():
        print(f"  WARNING: missing {path}", file=sys.stderr)
        continue
    data = load_json(path)
    label_key = next(iter(data))
    result = data[label_key]
    if buckets is None:
        buckets = [b for b, r in result.items() if r["total"] > 0]
    row = [model] + [
        f"{result.get(b, {}).get('accuracy', 0):.3f}"
        for b in buckets
    ]
    model_rows.append(row)
if buckets and model_rows:
    print_table("LongMemEval — accuracy by turn depth", model_rows, buckets)

print("\nDone.")
PYEOF

echo "Results written to $RESULTS_DIR"
