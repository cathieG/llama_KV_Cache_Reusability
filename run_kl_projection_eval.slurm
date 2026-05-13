#!/bin/bash
#SBATCH --job-name=kv-kl-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kv_kl_eval_%j.out
#SBATCH --error=logs/kv_kl_eval_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"
cd "$PROJECT_DIR"
mkdir -p logs outputs

echo "Job started on $(hostname)"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python scripts/run_transfer_learned.py \
  --checkpoint checkpoints/kv_projection_kl.pt \
  --output_file outputs/transfer_learned_projection_kl.jsonl

python scripts/compute_metrics.py \
  --native_file outputs/native_8b.jsonl \
  --transfer_file outputs/transfer_learned_projection_kl.jsonl \
  --output_file outputs/metrics_learned_projection_kl.json

echo "Job finished at $(date)"
