#!/bin/bash
#SBATCH --job-name=retest_15min3_fixed
#SBATCH --account=PAS2598
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=retest_15min3_fixed_%j.out
#SBATCH --error=retest_15min3_fixed_%j.err

module load cuda/12.6.2

cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Re-test the problematic image with the fix
python scripts/retest_15min3_fixed.py

echo "Job completed at $(date)"
