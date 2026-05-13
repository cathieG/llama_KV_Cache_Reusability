#!/bin/bash
#SBATCH --job-name=kv-layer-kl-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kv_layer_kl_eval_%j.out
#SBATCH --error=logs/kv_layer_kl_eval_%j.err

set -euo pipefail
PROJECT_DIR="$HOME/llama_KV_Cache_Reusability"
cd "$PROJECT_DIR"
mkdir -p logs outputs

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Python: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python scripts/run_transfer_target_layer_aware.py \
    --checkpoint checkpoints/kv_target_layer_adapter_kl.pt \
    --output_file outputs/transfer_target_layer_aware_kl.jsonl \
    --method_name target_layer_aware_kl

python scripts/compute_metrics.py \
    --native_file outputs/native_8b.jsonl \
    --transfer_file outputs/transfer_target_layer_aware_kl.jsonl \
    --output_file outputs/metrics_target_layer_aware_kl.json

echo "Job finished at $(date)"
