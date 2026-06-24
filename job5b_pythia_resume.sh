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

export WANDB_DIR=$SCRATCH
export WANDB_PROJECT=cortex-gpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=runs/pythia-5b

# Keep only the 3 most recent checkpoints to save storage
ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | head -n -3 | xargs -r rm -rf

# Find the latest checkpoint to resume from
LATEST=$(ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "No checkpoint found in ${RUN_DIR}. Run job5b_pythia.sh first."
    exit 1
fi
echo "Resuming from: $LATEST"

PREV_TOKENS=$(python -c "import torch; ckpt=torch.load('${LATEST}/checkpoint.pt', weights_only=False); print(ckpt['total_tokens'])")
NEW_MAX_TOKENS=$((PREV_TOKENS + 1250000000))
echo "Tokens so far: $PREV_TOKENS | Training until: $NEW_MAX_TOKENS"

python train.py \
    --training_mode pythia \
    --model_name EleutherAI/pythia-160m \
    --mean_recurrence 1 \
    --curriculum_steps 0 \
    --max_tokens ${NEW_MAX_TOKENS} \
    --schedule_tokens 5_000_000_000 \
    --batch_size 32768 \
    --micro_batch_size 4 \
    --out_dir ${RUN_DIR} \
    --resume_path ${LATEST} \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 2000
