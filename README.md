# 🔬 ColokRoll

**Colocalization analysis toolkit for fluorescence microscopy images**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ColokRoll is a Python toolkit designed for analyzing colocalization in multi-channel fluorescence microscopy images. It provides a complete pipeline from image loading to cell segmentation and colocalization quantification.

---

## Features

### Data Processing
- **Multi-format support**: Load images from `.nd2` (Nikon), `.oir` (Olympus), `.ome.tiff`, and standard TIFF formats
- **MIP Creation**: Maximum Intensity Projection with multiple methods
- **SME Projection**: Surface Manifold Extraction for 2.5D object handling with improved signal preservation

### Image Preprocessing
- **Z-slice detection**: Automatically identify and filter out-of-focus slices using focus metrics (Laplacian, Tenengrad, FFT)
- **Background subtraction**: Rolling ball algorithm with optional GPU acceleration (CuPy)
- **Quality control**: Focus quality metrics and automated slice selection

### Analysis
- **Cell segmentation**: Integration with [Cellpose](https://www.cellpose.org/) via HuggingFace Spaces API
- **Nuclei detection**: Automated nuclei identification
- **Colocalization metrics**: Pearson, Manders, and intensity correlation analysis

### 📊 Visualization
- Multi-channel composite generation
- Segmentation overlays with cell boundaries
- Z-slice focus score plots
- SME manifold visualization

---

## 🚀 Installation

### Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/SaezAtienzar/colok-roll.git
cd colok-roll

# Install in development mode
pip install -e .
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/SaezAtienzar/colok-roll.git
cd colok-roll

# Create environment from file
conda env create -f environment.yml
conda activate colok-roll

# Install the package
pip install -e .
```

### Optional dependencies

```bash
# For deep learning features (StarDist, etc.)
pip install -e ".[ml]"

# For GPU acceleration (requires CUDA)
pip install -e ".[gpu]"

# Full installation
pip install -e ".[full]"

# Development tools
pip install -e ".[dev]"
```

---

## 📖 Quick Start

### Basic Image Loading and Projection

```python
from colokroll.data_processing import ImageLoader, MIPCreator

# Load a microscopy image
loader = ImageLoader()
image = loader.load_image("path/to/image.x")

print(f"Image shape: {image.shape}")  # (Z, Y, X, C)
print(f"Channels: {loader.get_channel_names()}")
print(f"Pixel size: {loader.get_pixel_size()} μm")

# Create Maximum Intensity Projection
mip_creator = MIPCreator()
mip = mip_creator.create_mip(image, method="max")
```

### Z-Slice Filtering + SME Projection

```python
from colokroll.imaging_preprocessing import select_z_slices
from colokroll.data_processing import MIPCreator

# Filter out-of-focus slices
result = select_z_slices(
    image,
    method="combined",      # Focus metric
    strategy="relative",    # Detection strategy
    threshold=0.6,          # Quality threshold
)

# Keep only in-focus slices
filtered_image = image[result.indices_keep]
print(f"Kept {len(result.indices_keep)}/{image.shape[0]} slices")

# Create SME projection for better signal preservation
mip_creator = MIPCreator()
sme_result = mip_creator.create_sme(
    filtered_image,
    reference_channel=1,    # Channel for manifold computation
)

projection = sme_result.projection  # (Y, X, C)
manifold = sme_result.manifold      # Optimal Z per pixel
```

### Cell Segmentation with Cellpose

```python
from colokroll.analysis.cell_segmentation import CellSegmenter

# Initialize segmenter
segmenter = CellSegmenter(
    output_dir="/path/to/output",
)

# Segment cells from image array
result = segmenter.segment_from_image_array(
    image=filtered_image,
    channel_indices=(1, 3),      # (cell body, nuclei)
    channel_weights=(0.8, 0.2),  # Composite weights
)

print(f"Detected {result.mask_array.max()} cells")
print(f"Mask saved to: {result.mask_path}")
```

---

## Module Overview

```
colokroll/
├── core/                    # Configuration, utilities, format conversion
├── data_processing/         # Image loading and projection methods
│   ├── image_loader.py      # Multi-format image loader
│   └── projection.py        # MIP and SME projection
├── imaging_preprocessing/   # Image preprocessing pipeline
│   ├── z_slice_detection.py # Focus-based slice filtering
│   └── background_subtraction/
├── analysis/                # Analysis algorithms
│   ├── cell_segmentation.py # Cellpose integration
│   ├── nuclei_detection.py  # Nuclei identification
│   └── colocalization.py    # Colocalization metrics
└── visualization/           # Plotting and visualization tools
```

---

## Configuration

### Cellpose API Setup

Cell segmentation uses the HuggingFace Cellpose Space. Set your token:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

Or create a `.env` file in your project directory.

### GPU Acceleration

For GPU-accelerated background subtraction, install CuPy:

```bash
pip install cupy-cuda12x  # For CUDA 12.x
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation
- [BioIO](https://github.com/bioio-devs/bioio) for microscopy file format support
- [scikit-image](https://scikit-image.org/) for image processing algorithms

---

## Contact

**SaezAtienzar Lab**

- GitHub: [@SaezAtienzar](https://github.com/SaezAtienzar)
- Repository: [colok-roll](https://github.com/SaezAtienzar/colok-roll)

