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

# ── Easy / small-model benchmark suite ─────────────────────────────────────
# LAMBADA, BLIMP, SciQ, ARC-Easy, PIQA — the benchmarks that give usable signal
# at ~155M scale (HellaSwag / ARC-Challenge / GSM8K are at chance there; use
# job_eval_basic.sh for those).
export RESULTS_DIR="eval_results/easy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Model run directories.  The latest checkpoint in each is auto-discovered, so
# this keeps working as the runs advance (no hardcoded step numbers).
declare -A RUN_DIR=(
    ["pythia-5b"]="runs/pythia-5b"      # TRUE non-recurrent transformer baseline
    ["parcae-5b"]="runs/parcae-5b"      # Pre/Loop/Coda + LTI, no carry
    ["ccot-5b"]="runs/ccot-5b"          # Coconut-style full-network continuous CoT
    ["cortex-5b"]="runs/cortex-5b"      # Cortex K=0 (DirectCCoT carry)
    ["cortex-5b-k4"]="runs/cortex-5b-k4"  # Cortex K=4 (LM2 memory slots)
)
# T=1 for the non-recurrent pythia baseline; the recurrent / CCoT models omit
# --T so the eval CLI uses each checkpoint's saved mean_recurrence.
declare -A T_FLAG=(
    ["pythia-5b"]="--T 1"
    ["parcae-5b"]=""
    ["ccot-5b"]=""
    ["cortex-5b"]=""
    ["cortex-5b-k4"]=""
)

MODELS=("pythia-5b" "parcae-5b" "ccot-5b" "cortex-5b" "cortex-5b-k4")

echo "============================================================"
echo "Easy evals: LAMBADA, BLIMP, SciQ, ARC-Easy, PIQA"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    DIR="${RUN_DIR[$MODEL]}"
    # Pick the most recent checkpoint in this run directory.
    LATEST=$(ls -d ${DIR}/checkpoint_* 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST" ]; then
        echo "[$MODEL] no checkpoint found in ${DIR} — skipping"
        continue
    fi
    echo "[$MODEL] $LATEST/checkpoint.pt"
    python evals/eval_easy.py \
        --checkpoint "${LATEST}/checkpoint.pt" \
        ${T_FLAG[$MODEL]} \
        --tasks lambada blimp sciq arc_easy piqa \
        --out_dir "$RESULTS_DIR/$MODEL"
done

# ── Aggregate into a table ──────────────────────────────────────────────────
echo "============================================================"
echo "Results"
echo "============================================================"

python - <<'PYEOF'
import json, os, sys
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
MODELS = ["pythia-5b", "parcae-5b", "ccot-5b", "cortex-5b", "cortex-5b-k4"]
TASKS  = ["lambada", "blimp", "sciq", "arc_easy", "piqa"]

def load_json(path):
    if not path.exists():
        print(f"  WARNING: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

col_w   = 12
label_w = max(len(m) for m in MODELS) + 2
header  = f"{'Model':<{label_w}}" + "".join(f"{t:>{col_w}}" for t in TASKS)
sep     = "-" * len(header)
print("\nEasy benchmarks — accuracy")
print(sep); print(header); print(sep)
for m in MODELS:
    d = load_json(results_dir / m / "results.json")
    vals = []
    for t in TASKS:
        vals.append(f"{d[t]['accuracy']:.4f}" if (d and t in d) else "N/A")
    print(f"{m:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in vals))
print(sep)
print("\nDone.")
PYEOF

echo "Results written to $RESULTS_DIR"
