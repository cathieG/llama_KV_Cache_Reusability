#!/bin/bash
#SBATCH --job-name=kv-layer-mse
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kv_layer_mse_%j.out
#SBATCH --error=logs/kv_layer_mse_%j.err

set -euo pipefail
PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python scripts/train_target_layer_adapter.py \
    --data_file data/kv_pairs_by_target_layer.pt \
    --output_dir checkpoints \
    --output_name kv_target_layer_adapter.pt \
    --hidden_dim 256 \
    --epochs 5 \
    --batch_size 4096 \
    --lr 1e-3 \
    --weight_decay 1e-4

echo "Job finished at $(date)"
