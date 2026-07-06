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

# ── Easy evals, single model: cortex-5b-k4 ──────────────────────────────────
# One model per job so BLIMP (67k items) at T=8 fits the 24h walltime — the
# combined job_eval_easy.sh timed out after 2 of 5 models (job 10795331).
#
# All five per-model jobs submitted with the same EVAL_TAG write into one
# shared results root, so the combined table can be built afterwards:
#   EVAL_TAG=$(date +%Y%m%d) sbatch job_eval_easy_cortex_k4.sh
# (or just use submit_eval_easy_all.sh, which sets the tag once for all five).
# Without EVAL_TAG the tag defaults to the date the job STARTS, so jobs that
# come off the queue on different days would land in different roots.
MODEL="cortex-5b-k4"
RUN_DIR="runs/cortex-5b-k4"  # Cortex K=4 (LM2 memory slots)
T_FLAG=""                    # use the checkpoint's saved mean_recurrence (T=8)

RESULTS_ROOT="eval_results/easy_${EVAL_TAG:-$(date +%Y%m%d)}"
mkdir -p "$RESULTS_ROOT"

# Pick the most recent checkpoint (no hardcoded step numbers).
LATEST=$(ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "[$MODEL] no checkpoint found in ${RUN_DIR}"
    exit 1
fi

echo "============================================================"
echo "Easy evals [$MODEL]: LAMBADA, BLIMP, SciQ, ARC-Easy, PIQA"
echo "============================================================"
echo "[$MODEL] $LATEST/checkpoint.pt"
python evals/eval_easy.py \
    --checkpoint "${LATEST}/checkpoint.pt" \
    $T_FLAG \
    --tasks lambada blimp sciq arc_easy piqa \
    --out_dir "$RESULTS_ROOT/$MODEL"

echo "Results written to $RESULTS_ROOT/$MODEL"

# Combined table across whatever models have finished in this root so far;
# the last job to complete prints the full 5-model table.
python evals/aggregate_easy.py "$RESULTS_ROOT"
