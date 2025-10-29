#!/bin/bash
#SBATCH --job-name=oir-convert
#SBATCH --account=PAS2598
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=oir_conversion_%j.out
#SBATCH --error=oir_conversion_%j.err

# SLURM job script for batch OIR to OME-TIFF conversion
# 
# Usage:
#   sbatch run_oir_conversion.sh /path/to/oir/files
#
# Or edit INPUT_DIR below and run:
#   sbatch run_oir_conversion.sh

# ============================================================================
# Configuration
# ============================================================================

# Input directory containing .oir files (override with command line argument)
INPUT_DIR="${1:-/fs/scratch/PAS2598/duarte63/ALIX_confocal_data/Madi/Oct_12}"

# Script location
SCRIPT_DIR="/users/PAS2598/duarte63/GitHub/colok-roll"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_oir_only.py"

# ============================================================================
# Job Execution
# ============================================================================

echo "=========================================="
echo "OIR to OME-TIFF Batch Conversion"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Input Directory: ${INPUT_DIR}"
echo "=========================================="
echo ""

# Load conda
source ~/.bashrc

# Activate conversion environment
echo "Activating oir-convert environment..."
conda activate oir_convert

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate oir-convert environment"
    echo "Run setup first: ${SCRIPT_DIR}/setup_oir_conversion.sh"
    exit 1
fi

# Verify tools are available
echo "Checking conversion tools..."
which bioformats2raw || { echo "ERROR: bioformats2raw not found"; exit 1; }
which raw2ometiff || { echo "ERROR: raw2ometiff not found"; exit 1; }
echo "✓ Tools found"
echo ""

# Check input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Count .oir files
OIR_COUNT=$(find "${INPUT_DIR}" -maxdepth 1 -name "*.oir" -type f | wc -l)
echo "Found ${OIR_COUNT} .oir files to convert"
echo ""

if [ ${OIR_COUNT} -eq 0 ]; then
    echo "WARNING: No .oir files found in ${INPUT_DIR}"
    exit 0
fi

# Run conversion
echo "Starting conversion..."
echo "=========================================="
python "${CONVERT_SCRIPT}" "${INPUT_DIR}"
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Show summary of output files
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Output files created:"
    ls -lh "${INPUT_DIR}"/*.ome.tiff 2>/dev/null | wc -l
    echo "OME-TIFF files in ${INPUT_DIR}"
fi

# Deactivate conda environment
conda deactivate

exit ${EXIT_CODE}

