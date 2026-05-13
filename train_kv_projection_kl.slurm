#!/bin/bash
#SBATCH --job-name=kv-kl-proj
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kv_kl_projection_%j.out
#SBATCH --error=logs/kv_kl_projection_%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

# ---------------------------
# Project setup
# ---------------------------
# Change this if your project directory is different.
PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints outputs

# Optional: load CUDA if your cluster requires modules.
# Uncomment or modify if needed.
# module load cuda

# ---------------------------
# Environment check
# ---------------------------
echo "Python: $(which python)"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------
# KL-aligned projection training
# ---------------------------
python scripts/train_kv_projection_kl.py \
    --prompt_file prompts/prompts.jsonl \
    --init_checkpoint checkpoints/kv_projection.pt \
    --output_dir checkpoints \
    --output_name kv_projection_kl.pt \
    --epochs 1 \
    --kl_steps 5 \
    --lr 1e-5 \
    --temperature 1.0 \
    --kv_loss_weight 0.0 \
    --max_prompts 80 \
    --log_every 5 \
    --empty_cache_every 10

echo "Job finished at $(date)"
