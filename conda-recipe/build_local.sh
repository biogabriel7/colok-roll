#!/bin/bash
# Script to build and test the conda package locally before publishing

set -e

echo "🔨 Building colok-roll conda package locally..."
echo ""

# Check if conda-build is installed
if ! command -v conda-build &> /dev/null; then
    echo "❌ conda-build not found. Installing..."
    conda install -y conda-build
fi

# Build the package
echo "Building package..."
conda build conda-recipe/

# Get the path to the built package
PACKAGE_PATH=$(conda build conda-recipe/ --output)
echo ""
echo "✅ Package built successfully!"
echo "📦 Location: $PACKAGE_PATH"
echo ""

# Offer to install in a test environment
echo "To test the package in a fresh environment, run:"
echo ""
echo "  conda create -n test-colok-roll python=3.13"
echo "  conda activate test-colok-roll"
echo "  conda install --use-local colok-roll"
echo "  python test_installation.py"
echo ""
echo "To upload to anaconda.org, run:"
echo "  anaconda login"
echo "  anaconda upload $PACKAGE_PATH"

