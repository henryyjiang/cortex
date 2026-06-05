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

RUN_DIR=runs/cortex-5b

# Keep only the 3 most recent checkpoints to save storage
ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | head -n -3 | xargs -r rm -rf

# Find the latest checkpoint to resume from
LATEST=$(ls -d ${RUN_DIR}/checkpoint_* 2>/dev/null | sort | tail -1)
if [ -z "$LATEST" ]; then
    echo "No checkpoint found in ${RUN_DIR}. Run job5b.sh first."
    exit 1
fi
echo "Resuming from: $LATEST"

python train.py \
    --training_mode cortex \
    --model_name EleutherAI/pythia-160m \
    --mean_recurrence 8 \
    --max_tokens 1_250_000_000 \
    --batch_size 32768 \
    --micro_batch_size 4 \
    --curriculum_steps 3800 \
    --out_dir ${RUN_DIR} \
    --resume_path ${LATEST} \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 10000
