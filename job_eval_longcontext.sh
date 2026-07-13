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

export RESULTS_DIR="eval_results/longcontext_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Pre-downloaded dataset (no internet on compute nodes).
# Run once on a login node: python evals/download_datasets.py
LONGMEMEVAL_PATH="data/LongMemEval"
BABILONG_PATH="data/BABILong"

# Chunking matched to training: cortex-main trains with seq_len=2048
# sub-windows and cross_chunks=4 (auto for K>0 / ccot_direct), so
# NUM_CHUNKS=4 + SEQ_LEN=2048 reproduces the trained buffer regime
# (3 gated updates before the final read).  Multi-pass ablation knobs
# default off; see cortex-finetune/pace/eval_longcontext.sbatch for the
# FLOP-matched ladder rationale.
SEQ_LEN=${SEQ_LEN:-2048}
NUM_CHUNKS=${NUM_CHUNKS:-4}
PASSES_PER_CHUNK=${PASSES_PER_CHUNK:-1}
CCOT_PASSES=${CCOT_PASSES:-0}

# Second eval mode per model: NO-CHUNK — full context in one 2048-token
# window (the pythia-160m position limit; longer contexts fall back to
# 2048-token chunks) with 4 silent full passes carrying the buffer
# (latent multi-pass), then generate.  Results land in *-nochunk/ dirs.
# For models without cross state (pythia/parcae) the passes are no-ops —
# their no-chunk arm is the plain full-attention control.
SKIP_NOCHUNK=${SKIP_NOCHUNK:-false}
NOCHUNK_CCOT_PASSES=${NOCHUNK_CCOT_PASSES:-4}

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

# ── BABILong ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "BABILong"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    [ -z "${CKPTS[$MODEL]}" ] && continue
    echo "[$MODEL]"
    python evals/eval_babilong.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --tasks qa1 qa2 qa3 \
        --seq_len $SEQ_LEN \
        --num_chunks $NUM_CHUNKS \
        --passes_per_chunk $PASSES_PER_CHUNK \
        --ccot_passes $CCOT_PASSES \
        --dataset_path "$BABILONG_PATH" \
        --out_dir "$RESULTS_DIR/babilong/$MODEL"
    [ "$SKIP_NOCHUNK" = "true" ] || \
    python evals/eval_babilong.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --tasks qa1 qa2 qa3 \
        --seq_len 2048 \
        --num_chunks 1 \
        --ccot_passes $NOCHUNK_CCOT_PASSES \
        --dataset_path "$BABILONG_PATH" \
        --out_dir "$RESULTS_DIR/babilong-nochunk/$MODEL"
done

# ── LongMemEval ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "LongMemEval"
echo "============================================================"
for MODEL in "${MODELS[@]}"; do
    [ -z "${CKPTS[$MODEL]}" ] && continue
    echo "[$MODEL]"
    python evals/eval_longmemeval.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --seq_len $SEQ_LEN \
        --num_chunks $NUM_CHUNKS \
        --passes_per_chunk $PASSES_PER_CHUNK \
        --ccot_passes $CCOT_PASSES \
        --dataset_path "$LONGMEMEVAL_PATH" \
        --out_dir "$RESULTS_DIR/longmemeval/$MODEL"
    [ "$SKIP_NOCHUNK" = "true" ] || \
    python evals/eval_longmemeval.py \
        --checkpoint "${CKPTS[$MODEL]}" \
        ${T_FLAG[$MODEL]} \
        --seq_len 2048 \
        --num_chunks 1 \
        --ccot_passes $NOCHUNK_CCOT_PASSES \
        --dataset_path "$LONGMEMEVAL_PATH" \
        --out_dir "$RESULTS_DIR/longmemeval-nochunk/$MODEL"
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
    col_w   = max(len(h) for h in col_headers) + 2
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

# ── BABILong ──────────────────────────────────────────────────────────────────
for task in ["qa1", "qa2", "qa3"]:
    buckets = None
    rows = []
    for m in MODELS:
        d = load_json(results_dir / "babilong" / m / "results.json")
        if d is None:
            rows.append([m])
            continue
        label_key = next(iter(d))
        task_data = d[label_key].get(task, {})
        if buckets is None:
            buckets = [b for b, r in task_data.items() if r["total"] > 0]
        row = [m] + [f"{task_data.get(b, {}).get('accuracy', 0):.3f}" for b in (buckets or [])]
        rows.append(row)
    if buckets:
        print_table(f"BABILong {task.upper()} — accuracy by context length", rows, buckets)

# ── LongMemEval ───────────────────────────────────────────────────────────────
buckets = None
rows = []
for m in MODELS:
    d = load_json(results_dir / "longmemeval" / m / "results.json")
    if d is None:
        rows.append([m])
        continue
    label_key = next(iter(d))
    result = d[label_key]
    if buckets is None:
        buckets = [b for b, r in result.items() if r["total"] > 0]
    row = [m] + [f"{result.get(b, {}).get('accuracy', 0):.3f}" for b in (buckets or [])]
    rows.append(row)
if buckets:
    print_table("LongMemEval — accuracy by turn depth", rows, buckets)

print("\nDone.")
PYEOF

echo "Results written to $RESULTS_DIR"
