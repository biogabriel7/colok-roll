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
### Standard install
```bash
pip install colokroll
```
Installing the package pulls every runtime dependency required for the full pipeline (segmentation, quantification, visualization, and statistics). No extra phase-specific flags are necessary.

### Optional extras
Use extras only when you need additional tooling:

```bash
# CUDA support (pick the variant that matches your driver/toolkit)
pip install colokroll[cuda12]
pip install colokroll[cuda11]

# Run Cellpose via the Hugging Face Space client
pip install colokroll[space]

# Developer tooling (tests, formatting, docs)
pip install colokroll[dev]
```

### From source
```bash
git clone https://github.com/TheSaezAtienzarLab/colok-roll.git
cd colok-roll
pip install -e .[dev]
```

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
- numpy, scikit-image, matplotlib, Pillow, tifffile, aicsimageio
- nd2reader for Nikon pipelines
- cellpose, torch, opencv-python for segmentation
- scipy, pandas, seaborn, openpyxl, xlsxwriter for quantification and reporting
- gradio_client and PyYAML when the `space` extra is requested

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
