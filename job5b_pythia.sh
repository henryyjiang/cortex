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

# TRUE non-recurrent Pythia (GPTNeoX) transformer baseline.
# NOTE: this replaces the old `--training_mode vanilla` run, which trained the
# recurrent CortexGPT architecture at T=1 (with LTI + random h0) and was NOT a
# real transformer.  The arch differs, so start from a FRESH out_dir — old
# vanilla checkpoints in runs/pythia-5b cannot be loaded by the pythia arch.
python train.py \
    --training_mode pythia \
    --model_name EleutherAI/pythia-160m \
    --max_tokens 1_250_000_000 \
    --batch_size 32768 \
    --micro_batch_size 4 \
    --out_dir runs/pythia-5b \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 10000
