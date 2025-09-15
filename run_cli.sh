#!/bin/bash
#SBATCH --job-name=colokroll
#SBATCH --account={account}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --output=colokroll%j.out
#SBATCH --error=colokroll%j.err

module load cuda/12.6.2

source ~/.bashrc
conda activate colok-roll

# Optional: set working directory to repo root to make relative paths robust
cd /users/{account}/{user}/GitHub/colok-roll

echo "Job started on $(hostname) at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python scripts/batch_whole_analysis.py \
  --input-dir /fs/scratch/{account}/{user}/confocal_data/ \
  --patterns "*.ome.tiff" \ 
  --output-dir /fs/scratch/{account}/{user}/outputs \
  --log-level INFO

echo "Job completed at $(date)"