#!/bin/bash
#SBATCH --job-name=format_conv_all
#SBATCH --account=PAS2598
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=format_converter_all_%j.out
#SBATCH --error=format_converter_all_%j.err

# No GPU needed for format conversion

cd /users/PAS2598/duarte63/GitHub/colok-roll
source .venv/bin/activate

echo "=============================================="
echo "Format Converter - All Folders"
echo "=============================================="
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo ""

# Input directory with ND2 files (contains subfolders: 120min, 15min, 30min, 60min, ctrl)
INPUT_DIR="/fs/scratch/PAS2598/duarte63/confocal-images/bonetlab/ALIX"
OUTPUT_DIR="/fs/scratch/PAS2598/duarte63/outputs/format_converter_test"

echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Converting ALL files from all folders (15min, 30min, 60min, 120min, ctrl)..."
echo ""

# Run conversion for ALL files (no --max-files limit)
python scripts/test_format_converter.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --log-level INFO

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=============================================="

exit ${EXIT_CODE}
