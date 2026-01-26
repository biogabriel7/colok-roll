#!/bin/bash
#SBATCH --job-name=clorotoxin_pipeline
#SBATCH --account=PAS2598
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=clorotoxin_pipeline_%j.out
#SBATCH --error=clorotoxin_pipeline_%j.err

# GPU needed for puncta detection pipeline

cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "=============================================="
echo "Clorotoxin Puncta Pipeline"
echo "=============================================="
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo ""

# Input directory with ND2 files (contains subfolders: 15min, 30min, 60min, ctrl)
INPUT_DIR="/fs/scratch/PAS2598/duarte63/confocal-images/bonetlab/Clorotoxin"
# Output directory for puncta pipeline results
OUTPUT_DIR="/fs/scratch/PAS2598/duarte63/outputs/clorotoxin_puncta"

echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Dataset: Clorotoxin (45 .nd2 files across 4 time points)"
echo "Channels: GM130, Phalloidin, Chlorotoxin, DAPI"
echo "Z-slice strategy: FFT + Closest (Auto 0.8)"
echo ""

# Run puncta pipeline with auto-conversion of ND2 files
python scripts/test_clorotoxin_pipeline.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --auto-convert \
  --log-level INFO

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
