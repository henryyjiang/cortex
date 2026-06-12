#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 24:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex

export RESULTS_DIR="eval_results/basic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CORTEX_CKPT="runs/cortex-5b/checkpoint_0154441/checkpoint.pt"
CORTEX_K4_CKPT="runs/cortex-5b-k4/checkpoint_0152584/checkpoint.pt"
PYTHIA_CKPT="runs/pythia-5b/checkpoint_0152584/checkpoint.pt"
PARCAE_CKPT="runs/parcae-5b/checkpoint_0154441/checkpoint.pt"

declare -A CKPTS=(
    ["cortex-5b"]="$CORTEX_CKPT"
    ["cortex-5b-k4"]="$CORTEX_K4_CKPT"
    ["pythia-5b"]="$PYTHIA_CKPT"
    ["parcae-5b"]="$PARCAE_CKPT"
)
# memory_slots is read from each checkpoint's saved config by the eval CLIs.
# T=1 for pythia (vanilla transformer); None (omitted) for recurrent models
# so they use their saved mean_recurrence
declare -A T_FLAG=(
    ["cortex-5b"]=""
    ["cortex-5b-k4"]=""
    ["pythia-5b"]="--T 1"
    ["parcae-5b"]=""
)

MODELS=("cortex-5b" "cortex-5b-k4" "pythia-5b" "parcae-5b")

# ── Multiple choice (HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA) ───
echo "============================================================"
echo "Multiple choice"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    echo "[$MODEL]"
    python evals/eval_multiple_choice.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --tasks hellaswag winogrande arc_easy arc_challenge piqa \
        --out_dir "$RESULTS_DIR/multiple_choice/$MODEL"
done

# ── GSM8K ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "GSM8K"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    echo "[$MODEL]"
    python evals/eval_gsm8k.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --out_dir "$RESULTS_DIR/gsm8k/$MODEL"
done

# ── Aggregate into tables ──────────────────────────────────────────────────────
echo "============================================================"
echo "Results"
echo "============================================================"

python - <<'PYEOF'
import json, os, sys
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
MODELS = ["cortex-5b", "cortex-5b-k4", "pythia-5b", "parcae-5b"]

def load_json(path):
    if not path.exists():
        print(f"  WARNING: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

def print_table(title, rows, col_headers):
    col_w  = max(len(h) for h in col_headers) + 2
    label_w = max(len(r[0]) for r in rows) + 2
    header  = f"{'Model':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    sep     = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(f"{row[0]:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in row[1:]))
    print(sep)

# ── Multiple choice ───────────────────────────────────────────────────────────
MC_TASKS = ["hellaswag", "winogrande", "arc_easy", "arc_challenge", "piqa"]
rows = []
for m in MODELS:
    d = load_json(results_dir / "multiple_choice" / m / "results.json")
    vals = []
    for t in MC_TASKS:
        if d and t in d:
            vals.append(f"{d[t]['accuracy']:.4f}")
        else:
            vals.append("N/A")
    rows.append([m] + vals)
print_table("Multiple choice — accuracy", rows, MC_TASKS)

# ── GSM8K ─────────────────────────────────────────────────────────────────────
rows = []
for m in MODELS:
    d = load_json(results_dir / "gsm8k" / m / "results.json")
    acc = f"{d['accuracy']:.4f}" if d else "N/A"
    rows.append([m, acc])
print_table("GSM8K — accuracy (8-shot)", rows, ["Accuracy"])

print("\nDone.")
PYEOF

echo "Results written to $RESULTS_DIR"
