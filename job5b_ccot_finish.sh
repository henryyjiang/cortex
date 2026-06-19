#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
# CCoT is the heaviest variant; pin the 80GB A100 (verify the gres name on
# Phoenix with: sinfo -o '%G' | tr ',' '\n' | grep -i a100).
#SBATCH --gres=gpu:A100-80GB:1
#SBATCH -t 24:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=henryyjiang42@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex

export WANDB_DIR=$SCRATCH
export WANDB_PROJECT=cortex-gpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=runs/ccot-5b

# ── ONE-OFF: finish the first 1.25B-token chunk that timed out ~12 min short ──
# The normal resume script would set max_tokens = prev_tokens + 1.25B, which
# changes max_steps and therefore the whole LR/cooldown schedule.  Here we
# instead CAP at the ORIGINAL fresh-run budget (1.25B → max_steps 38146) so the
# schedule is byte-identical to the other runs and CCoT ends at the same step.
# After this completes, go back to the normal job5b_ccot_resume.sh for the next
# chunk (it will increment from the 1.25B checkpoint as usual).
#
# Resumes from the latest checkpoint (step 30000 — the last save before the
# timeout).  Steps 30000->38146 are re-run; that's expected.  DELETE this script
# once the chunk is done.
LATEST=$(ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "No checkpoint found in ${RUN_DIR}. Run job5b_ccot.sh first."
    exit 1
fi
echo "Resuming from: $LATEST (capping at 1.25B tokens to match the other runs)"

python train.py \
    --training_mode ccot \
    --model_name EleutherAI/pythia-160m \
    --mean_recurrence 8 \
    --max_tokens 1_250_000_000 \
    --batch_size 32768 \
    --micro_batch_size 4 \
    --grad_checkpoint \
    --curriculum_steps 3800 \
    --out_dir ${RUN_DIR} \
    --resume_path ${LATEST} \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 2000
