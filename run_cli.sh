#!/bin/bash
#SBATCH --job-name=bioimage_analysis
#SBATCH --account=PAS2598
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32gb
#SBATCH --output=bioimage_%j.out
#SBATCH --error=bioimage_%j.err

module load cuda/12.6.2

# Ensure conda is available in non-interactive shells
source ~/.bashrc
# or: source ~/miniconda3/etc/profile.d/conda.sh
conda activate bioimage_analyzer

# Optional: set working directory to repo root to make relative paths robust
cd /users/PAS2598/duarte63/GitHub/perinuclear_analysis

echo "Job started on $(hostname) at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python scripts/batch_whole_analysis.py \
  --input-dir /fs/scratch/PAS2598/duarte63/ALIX_confocal_data/ALIX/nd2 \
  --patterns "*.ome.tiff" \
  --output-dir /fs/scratch/PAS2598/duarte63/outputs \
  --log-level INFO

echo "Job completed at $(date)"