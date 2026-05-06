#!/bin/bash
#SBATCH --job-name=kv_learned
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

echo "Starting job on $(hostname)"
echo "Time: $(date)"

module load cuda

source ~/llama-kv-env/bin/activate

cd ~/llama_KV_Cache_Reusability

mkdir -p logs outputs data checkpoints

export TOKENIZERS_PARALLELISM=false

echo "Step 1: Collecting KV pairs"
python scripts/collect_kv_pairs.py \
  --output_file data/kv_pairs.pt \
  --max_prompts 80 \
  --samples_per_layer 64

echo "Step 2: Training learned projection"
python scripts/train_kv_projection.py \
  --data_file data/kv_pairs.pt \
  --output_dir checkpoints \
  --epochs 5 \
  --batch_size 4096 \
  --lr 1e-3

echo "Step 3: Running learned transfer"
python scripts/run_transfer_learned.py \
  --checkpoint checkpoints/kv_projection.pt \
  --output_file outputs/transfer_learned_projection.jsonl

echo "Finished at $(date)"
