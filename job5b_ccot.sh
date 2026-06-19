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

# NORMAL continuous chain-of-thought (Coconut-style) over the FULL transformer.
# Each recurrent pass runs the whole Pythia stack; the pass's last hidden state
# is added directly to the next pass's input embeddings (no loop block, no LTI,
# no projection).  Built on the pythia base.  Recurrence depth uses the same
# mean_recurrence / curriculum schedule as cortex / parcae for a fair compute-
# matched comparison.
#
# NOTE: each pass is a FULL-network forward, so T passes ≈ T× the per-token
# transformer compute (heavier than the cortex Loop block, which is ~50% of the
# layers).  If you hit OOM, lower --micro_batch_size (it must still divide
# batch_size/seq_len = 16).
python train.py \
    --training_mode ccot \
    --model_name EleutherAI/pythia-160m \
    --mean_recurrence 8 \
    --max_tokens 1_250_000_000 \
    --batch_size 32768 \
    --micro_batch_size 4 \
    --grad_checkpoint \
    --curriculum_steps 3800 \
    --out_dir runs/ccot-5b \
    --wandb_project cortex-gpt \
    --log_interval 10 \
    --save_interval 10000
