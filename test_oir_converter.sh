#!/bin/bash
#SBATCH --job-name=oir_test
#SBATCH --account=PAS2598
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=oir_test_%j.out
#SBATCH --error=oir_test_%j.err

# Load required modules
module load cuda/12.6.2

# Change to project directory
cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "Python version: $(python --version)"
echo "Testing OIR converter with bioio"

# Run the test script on sample OIR files
python scripts/test_format_converter.py \
  --input-dir /fs/scratch/PAS2598/duarte63/confocal-images/madi/ALIX/2025-09-18/non-targeting-control \
  --output-dir /fs/scratch/PAS2598/duarte63/outputs/oir_test \
  --max-files 2 \
  --log-level INFO

echo "Job completed at $(date)"
