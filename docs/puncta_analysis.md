# Puncta Analysis

Detect and analyze punctate structures (spots/vesicles) in fluorescence microscopy images.

---

## Overview

ColokRoll provides automated puncta detection with:

- **Two detection methods**: Laplacian of Gaussian (LoG) and BigFISH
- **Per-punctum metrics**: Area, intensity, SNR, nearest neighbor distance
- **Per-cell aggregation**: Puncta count, density, integrated intensity
- **Total image statistics**: Overall puncta counts and distributions

---

## Quick Start

```python
import colokroll as cr
import numpy as np

# Reconstruct 4D array from background subtraction results
corrected_stack = np.stack([
    results[ch][0] for ch in channel_names
], axis=-1)

# Detect puncta in LAMP1 channel
puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=mask_path,
    channel="LAMP1",
    channel_names=channel_names,
    pixel_size_um=0.108,
    detection_method="log",
)

# Access results
print(f"Total puncta: {puncta['results']['total_image']['total_puncta_count']}")
print(f"Cells analyzed: {puncta['results']['summary']['cells_count']}")
```

---

## Detection Methods

### Laplacian of Gaussian (LoG)

Default method using scikit-image. Best for general puncta detection:

```python
puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=mask_path,
    channel="LAMP1",
    channel_names=channel_names,
    detection_method="log",
    
    # LoG-specific parameters
    expected_diameter_um=0.4,    # Expected punctum size
    snr_threshold=3.0,           # Signal-to-noise threshold
    min_area_px=4,               # Minimum punctum area
)
```

### BigFISH

Automatic thresholding using the [BigFISH](https://github.com/fish-quant/big-fish) package:

```python
puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=mask_path,
    channel="LAMP1",
    channel_names=channel_names,
    detection_method="bigfish",
    
    # BigFISH uses automatic thresholding
    return_threshold_data=True,  # Get elbow curve data
)

# Access threshold information
threshold_data = puncta.get('threshold_data', {})
```

**Installation:**

```bash
pip install big-fish
```

---

## Parameters

### compute_puncta() Full Signature

```python
puncta = cr.compute_puncta(
    # Required
    image=corrected_stack,        # 4D array, path, or results dict
    mask=mask_path,               # Segmentation mask
    channel="LAMP1",              # Channel name or index
    
    # Channel identification
    channel_names=channel_names,  # Required if using channel names
    
    # Projection (for 3D data)
    projection="mip",             # "mip", "sme", or "none"
    
    # Detection method
    detection_method="log",       # "log" or "bigfish"
    
    # Physical parameters
    pixel_size_um=0.108,          # Pixel size in micrometers
    expected_diameter_um=0.4,     # Expected punctum diameter (µm)
    min_diameter_um=0.2,          # Minimum punctum diameter (µm)
    max_diameter_um=1.0,          # Maximum punctum diameter (µm)
    
    # Detection parameters
    snr_threshold=3.0,            # Signal-to-noise threshold
    min_distance_um=None,         # Min distance between puncta
    min_area_px=4,                # Minimum area in pixels
    max_area_px=None,             # Maximum area (auto from max_diameter)
    
    # Cell filtering
    drop_label_1=True,            # Remove Cellpose background label
    
    # Output
    return_threshold_data=False,  # Include BigFISH threshold data
    output_json="puncta.json",    # Save results to JSON
)
```

### Size Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `expected_diameter_um` | Expected punctum size | 0.3-0.5 µm for vesicles |
| `min_diameter_um` | Minimum allowed size | 0.2 µm |
| `max_diameter_um` | Maximum allowed size | 1.0-2.0 µm |
| `min_area_px` | Minimum area in pixels | 4-9 pixels |

### Detection Sensitivity

| Parameter | Effect |
|-----------|--------|
| `snr_threshold=2.0` | More puncta detected (more false positives) |
| `snr_threshold=3.0` | Balanced (default) |
| `snr_threshold=5.0` | Fewer puncta (only bright spots) |

---

## Results Structure

```python
puncta = cr.compute_puncta(...)

# Structure of results
puncta = {
    "image_shape": (14, 1800, 1800, 4),
    "channel": "LAMP1",
    "projection": "mip",
    "pixel_size_um": 0.108,
    "detection_params": {...},
    
    "results": {
        # Per-punctum data
        "puncta": [
            {
                "id": 1,
                "cell_label": 5,
                "centroid_y": 450.2,
                "centroid_x": 832.1,
                "area_px": 12,
                "area_um2": 0.14,
                "mean_intensity": 245.3,
                "integrated_intensity": 2943.6,
                "snr": 4.2,
                "nn_distance_px": 15.3,
                "nn_distance_um": 1.65,
            },
            ...
        ],
        
        # Per-cell aggregates
        "per_label": [
            {
                "label": 5,
                "cell_area_px": 45000,
                "cell_area_um2": 526.5,
                "puncta_count": 23,
                "puncta_density_per_px": 0.00051,
                "puncta_density_per_um2": 0.044,
                "total_integrated_intensity": 67890.0,
                "mean_puncta_area_px": 11.2,
                "mean_puncta_intensity": 231.5,
                "mean_puncta_snr": 3.8,
                "mean_nn_distance_um": 2.1,
            },
            ...
        ],
        
        # Total image metrics
        "total_image": {
            "total_cell_area_px": 1234567,
            "total_cell_area_um2": 14432.1,
            "total_puncta_count": 456,
            "total_puncta_density_per_px": 0.00037,
            "total_puncta_density_per_um2": 0.032,
            "total_integrated_intensity": 987654.0,
            "n_cells": 24,
        },
        
        # Summary (mean over cells)
        "summary": {
            "cells_count": 24,
            "mean_over_cells": {
                "puncta_count": 19.0,
                "puncta_density_per_um2": 0.036,
                "total_integrated_intensity": 41152.3,
                "mean_puncta_area_px": 10.8,
                "mean_puncta_intensity": 228.4,
                "mean_puncta_snr": 3.6,
                "mean_nn_distance_um": 2.3,
            },
        },
    },
}
```

---

## Working with Results

### Convert to DataFrame

```python
import pandas as pd

# Per-punctum data
df_puncta = pd.DataFrame(puncta['results']['puncta'])
print(f"Detected {len(df_puncta)} puncta")

# Per-cell data
df_cells = pd.DataFrame(puncta['results']['per_label'])
print(f"Across {len(df_cells)} cells")

# Summary
summary = puncta['results']['summary']['mean_over_cells']
print(f"Mean puncta per cell: {summary['puncta_count']:.1f}")
```

### Filter Puncta

```python
# Get only bright puncta
bright_puncta = [p for p in puncta['results']['puncta'] if p['snr'] > 5.0]

# Get puncta from specific cell
cell_5_puncta = [p for p in puncta['results']['puncta'] if p['cell_label'] == 5]
```

### Save Results

```python
# Save to JSON
puncta = cr.compute_puncta(
    ...,
    output_json="puncta_results.json",
)

# Save per-cell to CSV
df_cells = pd.DataFrame(puncta['results']['per_label'])
df_cells.to_csv("puncta_per_cell.csv", index=False)
```

---

## Input Formats

### From Background Subtraction Results

```python
# Use results dict directly
puncta = cr.compute_puncta(
    image=results,  # Dict from BackgroundSubtractor
    mask=mask_path,
    channel="LAMP1",
)
```

### From NumPy Array

```python
# 4D array (Z, Y, X, C)
puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=mask_path,
    channel=3,  # Channel index
)

# With channel names
puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=mask_path,
    channel="LAMP1",
    channel_names=['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'],
)
```

### From File Path

```python
puncta = cr.compute_puncta(
    image="path/to/image.ome.tiff",
    mask="path/to/mask.tif",
    channel=3,
)
```

---

## Complete Example

```python
import colokroll as cr
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load and preprocess
loader = cr.ImageLoader()
image = loader.load_image("sample.ome.tiff")
loader.rename_channels(['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'])
channel_names = loader.get_channel_names()
pixel_size = loader.get_pixel_size()

# 2. Z-slice selection
result = cr.select_z_slices(image, strategy="closest", keep_top=14)
filtered_image = image[result.indices_keep]

# 3. Background subtraction
bg_subtractor = cr.BackgroundSubtractor()
results = {}
for i, ch in enumerate(channel_names):
    corrected, meta = bg_subtractor.subtract_background(
        image=filtered_image[:, :, :, i],
        channel_name=ch,
    )
    results[ch] = (corrected, meta)

# 4. Segmentation
segmenter = cr.CellSegmenter(output_dir=Path("./output"))
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
)

# 5. Puncta analysis
corrected_stack = np.stack([results[ch][0] for ch in channel_names], axis=-1)

puncta = cr.compute_puncta(
    image=corrected_stack,
    mask=seg.mask_path,
    channel="LAMP1",
    channel_names=channel_names,
    pixel_size_um=pixel_size,
    detection_method="log",
    expected_diameter_um=0.4,
    snr_threshold=3.0,
    output_json="./output/puncta_results.json",
)

# 6. Analyze results
df_cells = pd.DataFrame(puncta['results']['per_label'])
total = puncta['results']['total_image']
summary = puncta['results']['summary']

print(f"Total puncta detected: {total['total_puncta_count']}")
print(f"Cells analyzed: {summary['cells_count']}")
print(f"Mean puncta per cell: {summary['mean_over_cells']['puncta_count']:.1f}")
print(f"Mean density (per µm²): {summary['mean_over_cells']['puncta_density_per_um2']:.4f}")

# Save per-cell results
df_cells.to_csv("./output/puncta_per_cell.csv", index=False)
```

---

## Tips and Troubleshooting

### Too Many Puncta Detected

- Increase `snr_threshold` (e.g., 4.0 or 5.0)
- Increase `min_area_px`
- Check background subtraction quality

### Too Few Puncta Detected

- Decrease `snr_threshold` (e.g., 2.0)
- Check that `expected_diameter_um` matches your puncta size
- Decrease `min_area_px`

### BigFISH Not Working

```bash
# Install BigFISH
pip install big-fish

# Verify installation
python -c "from bigfish import detection; print('OK')"
```

### Memory Issues with Large Images

```python
# Use MIP projection to reduce data
puncta = cr.compute_puncta(
    image=image,
    mask=mask,
    channel="LAMP1",
    projection="mip",  # Project to 2D first
)
```

### Validating Detection

```python
# Return threshold data for inspection
puncta = cr.compute_puncta(
    ...,
    detection_method="bigfish",
    return_threshold_data=True,
)

# Inspect threshold curve
if 'threshold_data' in puncta:
    print(puncta['threshold_data'])
```

