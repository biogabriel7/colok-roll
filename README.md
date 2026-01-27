# ColokRoll

**Colocalization analysis for fluorescence microscopy images**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ColokRoll is a Python toolkit for analyzing colocalization in multi-channel fluorescence microscopy images. It handles image loading, cell segmentation, colocalization quantification, and puncta analysis.

---

## Features

| Module | Description |
|--------|-------------|
| Image Loading | Multi-format support (`.nd2`, `.oir`, `.ome.tiff`, TIFF) |
| Z-Slice Detection | Focus-based filtering with multiple strategies |
| Background Subtraction | GPU-accelerated with negative control support |
| Cell Segmentation | Cellpose integration via HuggingFace Spaces |
| Colocalization | Pearson, Manders, Jaccard metrics per-cell |
| Puncta Analysis | LoG and BigFISH detection methods |

---

## Workflow modes

ColokRoll supports two workflow modes for processing microscopy data:

### Exploratory mode (parameter calibration)

Use on your first image to visually inspect and select optimal parameters:

```python
import colokroll as cr

# Compare Z-slice detection strategies
comparison = cr.compare_strategies(image, display_inline=True)

# Visually inspect and pick the best strategy
result = comparison.results["FFT + Closest (k=14)"]
filtered_image = image[result.indices_keep]

# Calibrate background subtraction with negative control
corrected, meta = bg_subtractor.subtract_background(
    image=channel_data,
    channel_name="ALIX",
    is_negative_control=True,  # Optimize for minimal residual signal
)
# Extract validated parameters for batch processing
best_params = meta['parameters_used']
```

### Batch mode (production processing)

Apply validated parameters consistently across all images:

```python
# Use explicit parameters from calibration
result = cr.select_z_slices(image, method="fft", strategy="closest", keep_top=14)

# Apply validated background subtraction parameters
corrected, meta = bg_subtractor.subtract_background(
    image=channel_data,
    method="two_stage",
    **validated_params  # From negative control calibration
)
```

See [docs/workflow_modes.md](docs/workflow_modes.md) for detailed guidance.

---

## Installation

```bash
# Clone and install
git clone https://github.com/SaezAtienzar/colok-roll.git
cd colok-roll
pip install -e .

# With GPU acceleration
pip install -e ".[gpu]"
```

---

## Quick start

```python
import colokroll as cr
from pathlib import Path

# 1. Load image
loader = cr.ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")
loader.rename_channels(['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'])

# 2. Z-slice selection
result = cr.select_z_slices(image, method="combined", strategy="relative", threshold=0.6)
filtered_image = image[result.indices_keep]

# 3. Background subtraction
bg_subtractor = cr.BackgroundSubtractor()
results = {}
for i, ch in enumerate(loader.get_channel_names()):
    corrected, meta = bg_subtractor.subtract_background(
        image=filtered_image[:, :, :, i],
        channel_name=ch,
        is_negative_control=(ch == "ALIX"),  # If this is a negative control
    )
    results[ch] = (corrected, meta)

# 4. Cell segmentation
segmenter = cr.CellSegmenter(output_dir=Path("./output"))
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
)

# 5. Colocalization analysis
import numpy as np
corrected_stack = np.stack([results[ch][0] for ch in loader.get_channel_names()], axis=-1)

coloc = cr.compute_colocalization(
    image=corrected_stack,
    mask=seg.mask_path,
    channel_a="ALIX",
    channel_b="LAMP1",
    channel_names=loader.get_channel_names(),
    thresholding="otsu",
)
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Workflow modes](docs/workflow_modes.md) | Exploratory vs batch processing |
| [Z-slice detection](docs/z_slice_detection.md) | Focus metrics and strategy comparison |
| [Background subtraction](docs/background_subtraction.md) | Methods and negative control support |
| [Cell segmentation](docs/cell_segmentation.md) | Cellpose integration |
| [Colocalization](docs/colocalization.md) | Metrics and analysis |
| [Puncta analysis](docs/puncta_analysis.md) | Spot detection with BigFISH |

---

## Module overview

```
colokroll/
├── core/                    # Configuration, utilities
├── data_processing/         # Image loading, projections (MIP, SME)
├── imaging_preprocessing/   # Z-slice detection, background subtraction
├── analysis/                # Segmentation, colocalization, puncta
└── visualization/           # Plotting tools
```

---

## Configuration

### GPU acceleration

```bash
pip install cupy-cuda12x  # For CUDA 12.x
```

### Cellpose API

Cell segmentation uses HuggingFace Cellpose Space (no local installation required).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation
- [BigFISH](https://github.com/fish-quant/big-fish) for puncta detection
- [BioIO](https://github.com/bioio-devs/bioio) for microscopy format support
- [scikit-image](https://scikit-image.org/) for image processing

---

**SaezAtienzar Lab** | [GitHub](https://github.com/SaezAtienzar/colok-roll)
