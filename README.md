# colok&roll

A comprehensive Python library for analyzing signals in confocal microscopy images. This library provides a phased implementation approach for processing 3D z-stack images, performing cell segmentation, background subtraction, and colocalization analysis.

## Features

### **Core Functionality**
- **3D Image Processing**: Load and process whatever file format confocal microscopy images 
- **Cell Segmentation**: Automated cell segmentation using Cellpose via Hugging Face Spaces
- **Background Subtraction**: CUDA-accelerated background subtraction
- **Colocalization Analysis**: Quantitative analysis of protein colocalization patterns
- **Multi-channel Support**: Handle complex multi-channel fluorescence images

### ⚡ **Performance Optimizations**
- **CUDA Acceleration**: GPU-accelerated processing
- **Memory Efficient**: Chunk-based processing optimized

### 🎯 **Research Applications**
- Protein colocalization studies
- Cell morphology analysis
- High-throughput microscopy data processing

## Installation

### Basic Installation

```bash
pip install colokroll
```

### Phase-specific Installation

Install specific functionality phases:

```bash
# Phase 1-2: Core functionality (image loading, MIP creation, visualization)
pip install colokroll[phase1,phase2]

# Phase 3: Cell segmentation
pip install colokroll[phase3]

# Phase 4: Ring analysis
pip install colokroll[phase4]

# Phase 5: Complete analysis pipeline
pip install colokroll[phase5]

# All features
pip install colokroll[all]
```

### CUDA Support (Optional)

For GPU acceleration, install CUDA dependencies:

```bash
# For CUDA 12.x
pip install colokroll[cuda12]

# For CUDA 11.x
pip install colokroll[cuda11]
```

### Development Installation

```bash
git clone https://github.com/TheSaezAtienzarLab/colok-roll.git
cd colok-roll
pip install -e .[dev]
```

## Quick Start

### Basic Image Analysis

```python
from colokroll import ImageLoader, CellSegmenter, BackgroundSubtractor

# Load image
loader = ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")

# Rename channels for clarity
channel_names = loader.rename_channels(['LAMP1', 'Phalloidin', 'ALIX', 'DAPI'])

# Background subtraction
bg_subtractor = BackgroundSubtractor()
processed_image = bg_subtractor.process(image, channel_names=['LAMP1', 'Phalloidin'])

# Cell segmentation
segmenter = CellSegmenter()
result = segmenter.segment_from_file(
    "path/to/image.ome.tiff",
    channel_names=channel_names,
    channel_a='Phalloidin',
    channel_b='DAPI'
)
```

### Batch Processing

Use the provided batch processing script:

```bash
python scripts/batch_whole_analysis.py \
  --input-dir /path/to/images \
  --output-dir /path/to/outputs \
  --patterns "*.ome.tiff"
```

### SLURM Cluster Usage

For HPC environments, use the provided SLURM script:

```bash
sbatch run_cli.sh
```

## Architecture

### Phase-based Design

The library is organized into 5 phases, each building upon the previous:

- **Phase 1**: Infrastructure & Image Loading
- **Phase 2**: MIP Creation & Basic Visualization  
- **Phase 3**: Cell & Nuclei Segmentation
- **Phase 4**: Ring Analysis
- **Phase 5**: Signal Quantification & Complete Pipeline

### Module Structure

```
perinuclear_analysis/
├── core/                    # Configuration and utilities
├── data_processing/         # Image loading and MIP creation
├── imaging_preprocessing/   # Background subtraction, denoising
├── analysis/               # Cell segmentation, colocalization
└── visualization/          # Plotting and visualization tools
```

## Configuration

### Preprocessing Templates

The library includes pre-configured templates for common analysis scenarios:

```python
from colokroll.core import create_preprocessing_templates

# Create analysis-specific templates
templates = create_preprocessing_templates()

# Available templates:
# - standard_4channel: General purpose DAPI/Phalloidin/LAMP1/GAL3
# - colocalization_optimized: Conservative parameters for quantitative accuracy
# - high_throughput: Speed/memory optimized for batch processing
# - quality_assessment: Maximum quality with detailed metrics
```

### Custom Configuration

```python
from colokroll.core import BackgroundSubtractionConfig

config = BackgroundSubtractionConfig(
    method="rolling_ball",
    radius=30,
    light_background=False
)

bg_subtractor = BackgroundSubtractor(config=config)
```

## Supported File Formats

- **OME-TIFF**: Primary format with full metadata support
- **ND2**: Nikon NIS-Elements files
- **TIFF**: Standard TIFF files
- **PNG/JPG**: For visualization outputs

## Requirements

### System Requirements
- Python 3.8+
- 16GB+ RAM (recommended for large 3D datasets)
- CUDA-compatible GPU (optional, for acceleration)

### Dependencies

**Core Dependencies:**
- numpy >= 1.20.0
- scikit-image >= 0.19.0
- matplotlib >= 3.5.0
- nd2reader >= 3.3.0
- tifffile >= 2023.7.10

**Optional Dependencies:**
- cellpose >= 2.0.0 (segmentation)
- torch >= 1.10.0 (ML models)
- cupy (CUDA acceleration)
- pandas, seaborn (data analysis)

## Examples

### Complete Analysis Pipeline

```python
import numpy as np
from colokroll import (
    ImageLoader, 
    BackgroundSubtractor, 
    CellSegmenter,
    compute_colocalization
)

# 1. Load and preprocess image
loader = ImageLoader()
image = loader.load_image("sample.ome.tiff")
channels = loader.rename_channels(['LAMP1', 'Phalloidin', 'ALIX', 'DAPI'])

# 2. Background subtraction
bg_subtractor = BackgroundSubtractor()
processed = bg_subtractor.process(image, channel_names=channels)

# 3. Cell segmentation
segmenter = CellSegmenter()
segmentation_result = segmenter.segment_from_file(
    "sample.ome.tiff",
    channel_names=channels,
    channel_a='Phalloidin',
    channel_b='DAPI'
)

# 4. Colocalization analysis
coloc_results = compute_colocalization(
    image=processed,
    channel_a='ALIX',
    channel_b='LAMP1',
    masks=segmentation_result.mask_array
)
```

### Visualization

```python
from colokroll.visualization import plot_mip, plot_channels

# Plot MIP projections
plot_mip(image, channels=['LAMP1', 'Phalloidin', 'ALIX', 'DAPI'])

# Plot individual channels
plot_channels(image, channel_names=channels)
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## Citation

If you use this library in your research, please cite:

```bibtex
@software{perinuclear_analysis,
  title={Perinuclear Analysis: A Python Library for Confocal Microscopy Image Analysis},
  author={Gabriel Duarte},
  year={2024},
  url={https://github.com/TheSaezAtienzarLab/colok-roll}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Wiki](https://github.com/TheSaezAtienzarLab/colok-roll/wiki)
- **Issues**: [GitHub Issues](https://github.com/TheSaezAtienzarLab/colok-roll/issues)
- **Email**: gabriel.duarte@osumc.edu

## Acknowledgments

- Built for the Saez-Atienzar Lab at The Ohio State University
- Utilizes Cellpose for cell segmentation
- CUDA acceleration powered by CuPy
- Image I/O supported by nd2reader and tifffile

---

**Note**: This library is in active development. Some features may be experimental or subject to change.
