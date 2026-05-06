#!/bin/bash
#SBATCH --job-name=kv_interp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

echo "Starting job on $(hostname)"
echo "Job Start Time: $(date)"

# Load modules (adjust if needed)
module load cuda

# Move to your project directory
cd ~/llama_KV_Cache_Reusability

# Run your script
python scripts/run_interpolation.py --method interpolation

echo "Finished at $(date)"
