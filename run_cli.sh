#!/bin/bash
#SBATCH --job-name=bioimage_analyzer
#SBATCH --account=PAS2598
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=bioimage_analyzer%j.out
#SBATCH --error=bioimage_analyzer%j.err

module load cuda/12.6.2

source ~/.bashrc
conda activate bioimage_analyzer

echo "Job started on $(hostname) at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python scripts/batch_whole_analysis.py \
  --input-dir /fs/scratch/PAS2598/duarte63/ALIX_confocal_data/Madi/Oct_28 \
  --patterns "*.ome.tiff" \ 
  --output-dir /fs/scratch/PAS2598/duarte63/outputs/Madi \
  --log-level INFO

echo "Job completed at $(date)"