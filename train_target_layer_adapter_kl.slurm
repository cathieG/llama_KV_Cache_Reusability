#!/bin/bash
#SBATCH --job-name=kv-layer-kl
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kv_layer_kl_%j.out
#SBATCH --error=logs/kv_layer_kl_%j.err

set -euo pipefail
PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python scripts/train_target_layer_adapter_kl.py \
    --prompt_file prompts/prompts.jsonl \
    --init_checkpoint checkpoints/kv_target_layer_adapter.pt \
    --output_dir checkpoints \
    --output_name kv_target_layer_adapter_kl.pt \
    --epochs 1 \
    --kl_steps 5 \
    --lr 1e-5 \
    --temperature 1.0 \
    --max_prompts 80 \
    --log_every 5 \
    --empty_cache_every 10

echo "Job finished at $(date)"
