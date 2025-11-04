# colok&roll

A Python toolkit for perinuclear and colocalization analysis in confocal microscopy images. The library bundles image loading, preprocessing, segmentation, quantitative analysis, and visualization utilities so you can run the entire workflow with a single installation.

## Features
- Robust image ingestion with automatic conversion from ND2 and OIR into OME-TIFF while preserving metadata
- Multi-channel 3D support, including maximum-intensity projections and per-channel utilities
- CUDA-accelerated background subtraction with automatic CPU fallbacks for non-GPU paths
- Cell and nuclei segmentation powered by Cellpose and Torch
- Ring analysis, signal quantification, and colocalization helpers for downstream statistics
- Ready-to-use visualization helpers for quick inspection of intermediate results

## Installation

### Conda Installation (Recommended)

The easiest way to install `colok-roll` with all dependencies is using conda:

```bash
# Clone the repository
git clone https://github.com/TheSaezAtienzarLab/colok-roll.git
cd colok-roll

# Create and activate the environment with all dependencies
conda env create -f environment.yml
conda activate colok-roll

# Install the package
pip install -e .
```

This single command installs **all dependencies** needed for the complete analysis pipeline:
- Image processing and I/O (numpy, scipy, scikit-image, opencv)
- Deep learning segmentation (PyTorch, TensorFlow, Cellpose, StarDist)
- GPU acceleration (CuPy with CUDA support)
- Data analysis and export (pandas, seaborn, openpyxl, xlsxwriter)
- Visualization (matplotlib, Jupyter)

See [CONDA_INSTALL.md](CONDA_INSTALL.md) for detailed installation instructions and troubleshooting.

### Pip Installation

```bash
pip install colokroll
```

Note: pip installation may require manual setup of CUDA and other system dependencies.

### Developer Installation

For development with testing and documentation tools:

```bash
conda activate colok-roll
pip install -e ".[dev]"
```

This includes: pytest, black, flake8, sphinx, and jupyterlab.

## Quick start
```python
from colokroll import ImageLoader, CellSegmenter
from colokroll.imaging_preprocessing.background_subtraction import BackgroundSubtractor
from colokroll.visualization import plot_mip

loader = ImageLoader()
image = loader.load_image("path/to/sample.oir")  # automatically converts to OME-TIFF
channel_names = loader.get_channel_names() or loader.rename_channels(["LAMP1", "Phalloidin", "ALIX", "DAPI"])

# Optional preprocessing (requires CUDA if using BackgroundSubtractor)
bg_subtractor = BackgroundSubtractor()
processed = bg_subtractor.subtract_background(image[..., 0], channel_name=channel_names[0])[0]

segmenter = CellSegmenter()
result = segmenter.segment_from_file(
    "path/to/sample.ome.tiff",
    channel_names=channel_names,
    channel_a="Phalloidin",
    channel_b="DAPI",
)

plot_mip(image, channel_names=channel_names)
```

## Workflow overview
1. Load ND2, OIR, or OME-TIFF files with automatic metadata extraction
2. (Optional) Apply preprocessing modules: background subtraction, denoising, deconvolution
3. Segment nuclei and cells, then build perinuclear rings
4. Quantify channel intensities and compute colocalization statistics
5. Visualize results or export tables for further analysis

## Module layout
```
colokroll/
├── core/                  # Configuration classes and shared utilities
├── data_processing/       # Image loaders, format conversion, MIP creation
├── imaging_preprocessing/ # Background subtraction and preprocessing helpers
├── analysis/              # Segmentation, colocalization, quantification routines
└── visualization/         # Visualization helpers
```

## Supported file formats
- OME-TIFF (native processing format)
- ND2 (Nikon) with conversion to OME-TIFF
- OIR (Olympus) with conversion to OME-TIFF
- Standard TIFF/PNG/JPG for masks and derived outputs

## Runtime dependencies
Installed alongside the package:
- numpy, scikit-image, matplotlib, Pillow, tifffile
- nd2reader for Nikon pipelines
- cellpose, stardist, pytorch, tensorflow for segmentation
- cupy for GPU acceleration
- scipy, pandas, seaborn, openpyxl, xlsxwriter for quantification and reporting
- gradio_client and PyYAML for remote processing

## Publishing

To publish this package to conda-forge for public distribution:

1. **Quick Start**: See [QUICK_PUBLISH_GUIDE.md](QUICK_PUBLISH_GUIDE.md)
2. **Detailed Guide**: See [PUBLISHING.md](PUBLISHING.md)

After publishing to conda-forge, users will be able to install with:
```bash
conda install -c conda-forge colok-roll
```

## Contributing
We welcome pull requests! Please open an issue to discuss large changes, follow the linting/test suite (`pytest`), and format code with `black` before submission.

## Citation
```bibtex
@software{colokroll,
  title={colok&roll: A Python Library for Confocal Microscopy Image Analysis},
  author={Gabriel Duarte},
  year={2024},
  url={https://github.com/TheSaezAtienzarLab/colok-roll}
}
```

## License
Released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support
- Issues: https://github.com/TheSaezAtienzarLab/colok-roll/issues
- Email: gabriel.duarte@osumc.edu
