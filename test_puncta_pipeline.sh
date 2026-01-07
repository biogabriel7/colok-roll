#!/bin/bash
#SBATCH --job-name=puncta_batch
#SBATCH --account=PAS2598
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=puncta_batch_%j.out
#SBATCH --error=puncta_batch_%j.err

module load cuda/12.6.2

cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Process all ome.tiff images in the input directory
python scripts/test_puncta_pipeline.py \
  --input-dir /fs/scratch/PAS2598/duarte63/ALIX_confocal_data/ALIX/nd2 \
  --output-dir /fs/scratch/PAS2598/duarte63/outputs/puncta_batch

echo "Job completed at $(date)"
