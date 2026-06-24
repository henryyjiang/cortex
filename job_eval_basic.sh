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

# ── Basic / harder benchmark suite ──────────────────────────────────────────
# Multiple choice (HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA) + GSM8K.
# These mostly need larger scale than 155M to clear chance; job_eval_easy.sh has
# the benchmarks that give usable signal at this size.

# Model run directories.  The latest checkpoint in each is auto-discovered, so
# this keeps working as the runs advance (no hardcoded step numbers).
declare -A RUN_DIR=(
    ["pythia-5b"]="runs/pythia-5b"        # TRUE non-recurrent transformer baseline
    ["parcae-5b"]="runs/parcae-5b"        # Pre/Loop/Coda + LTI, no carry
    ["ccot-5b"]="runs/ccot-5b"            # Coconut-style full-network continuous CoT
    ["cortex-5b"]="runs/cortex-5b"        # Cortex K=0 (DirectCCoT carry)
    ["cortex-5b-k4"]="runs/cortex-5b-k4"  # Cortex K=4 (LM2 memory slots)
)
# T=1 for the non-recurrent pythia baseline; the recurrent / CCoT models omit
# --T so the eval CLI uses each checkpoint's saved mean_recurrence.
# memory_slots is read from each checkpoint's saved config by the eval CLIs.
declare -A T_FLAG=(
    ["pythia-5b"]="--T 1"
    ["parcae-5b"]=""
    ["ccot-5b"]=""
    ["cortex-5b"]=""
    ["cortex-5b-k4"]=""
)

MODELS=("pythia-5b" "parcae-5b" "ccot-5b" "cortex-5b" "cortex-5b-k4")

# Resolve each model to its latest checkpoint once, up front.
declare -A CKPTS=()
for MODEL in "${MODELS[@]}"; do
    LATEST=$(ls -d ${RUN_DIR[$MODEL]}/checkpoint_* 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST" ]; then
        echo "WARNING: no checkpoint found in ${RUN_DIR[$MODEL]} — $MODEL will be skipped"
        continue
    fi
    CKPTS[$MODEL]="${LATEST}/checkpoint.pt"
    echo "[$MODEL] ${CKPTS[$MODEL]}"
done

# ── Multiple choice (HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA) ───
echo "============================================================"
echo "Multiple choice"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    [ -z "${CKPTS[$MODEL]}" ] && continue
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
    [ -z "${CKPTS[$MODEL]}" ] && continue
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
MODELS = ["pythia-5b", "parcae-5b", "ccot-5b", "cortex-5b", "cortex-5b-k4"]

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
