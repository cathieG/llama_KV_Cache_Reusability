#!/bin/bash
#SBATCH --job-name=kv-layer-collect
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kv_layer_collect_%j.out
#SBATCH --error=logs/kv_layer_collect_%j.err

set -euo pipefail
PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"
cd "$PROJECT_DIR"
mkdir -p logs data

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python scripts/collect_kv_pairs_by_target_layer.py \
    --prompt_file prompts/prompts.jsonl \
    --output_file data/kv_pairs_by_target_layer.pt \
    --max_prompts 80 \
    --samples_per_layer 64

echo "Job finished at $(date)"
