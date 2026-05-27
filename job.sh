#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100:1
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o logs/Report-%j.out

cd $SLURM_SUBMIT_DIR

module load anaconda3
module load cuda/12.1.1
conda activate cortex

export WANDB_DIR=$SCRATCH
export WANDB_PROJECT=cortex-gpt

python train.py \
    --training_mode retrofit \
    --model_name EleutherAI/pythia-160m \
    --mean_recurrence 8 \
    --max_tokens 500_000_000 \
    --batch_size 512 \
    --micro_batch_size 4 \
    --out_dir runs/retrofit-verify \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 500
