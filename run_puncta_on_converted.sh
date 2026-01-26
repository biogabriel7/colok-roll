#!/bin/bash
#SBATCH --job-name=puncta_converted
#SBATCH --account=PAS2598
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=puncta_converted_%j.out
#SBATCH --error=puncta_converted_%j.err

# GPU needed for puncta detection pipeline

cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "=============================================="
echo "Puncta Pipeline on Converted Files"
echo "=============================================="
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo ""

# Directory with converted OME-TIFF files (from format converter test)
CONVERTED_DIR="/fs/scratch/PAS2598/duarte63/outputs/format_converter_test"
# Original input directory (with folder structure: 15min, 30min, 60min, 120min, ctrl)
ORIGINAL_INPUT_DIR="/fs/scratch/PAS2598/duarte63/confocal-images/bonetlab/ALIX"
# New output directory for puncta pipeline results
OUTPUT_DIR="/fs/scratch/PAS2598/duarte63/outputs/puncta_pipeline_converted"

echo "Converted files directory: ${CONVERTED_DIR}"
echo "Original input directory: ${ORIGINAL_INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run puncta pipeline on all converted files
# --auto-convert will automatically convert any missing ND2/OIR files
python scripts/run_puncta_on_converted.py \
  --converted-dir "${CONVERTED_DIR}" \
  --original-input-dir "${ORIGINAL_INPUT_DIR}" \
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
