# Colocalization Analysis

Quantify spatial correlation between fluorescent markers using standard colocalization metrics.

---

## Overview

ColokRoll computes colocalization metrics between two channels:

- **Pearson correlation**: Linear correlation of intensities
- **Manders coefficients (M1/M2)**: Fraction of signal overlap
- **Jaccard index**: Overlap of thresholded regions
- Per-cell and whole-image metrics

---

## Quick Start

```python
import colokroll as cr
import numpy as np

# Reconstruct 4D array from background subtraction results
corrected_stack = np.stack([
    results[ch][0] for ch in channel_names
], axis=-1)

# Compute colocalization
res = cr.compute_colocalization(
    image=corrected_stack,
    mask=mask_path,               # Segmentation mask
    channel_a="ALIX",
    channel_b="LAMP1",
    channel_names=channel_names,
    thresholding="otsu",
)

# Access results
print(f"Pearson r: {res['results']['total_image']['pearson_r']:.3f}")
print(f"Manders M1: {res['results']['total_image']['manders_m1']:.3f}")
print(f"Manders M2: {res['results']['total_image']['manders_m2']:.3f}")
```

---

## Colocalization Metrics

### Pearson Correlation Coefficient

Measures linear correlation between channel intensities:

$$r = \frac{\sum(A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum(A_i - \bar{A})^2 \sum(B_i - \bar{B})^2}}$$

| Value | Interpretation |
|-------|----------------|
| +1.0 | Perfect positive correlation |
| 0 | No correlation |
| -1.0 | Perfect negative correlation |

```python
pearson = res['results']['total_image']['pearson_r']
```

### Manders Coefficients

Measure fractional overlap of signal:

- **M1**: Fraction of channel A that overlaps with channel B
- **M2**: Fraction of channel B that overlaps with channel A

$$M_1 = \frac{\sum A_i \cdot B_{mask,i}}{\sum A_i}$$

| Value | Interpretation |
|-------|----------------|
| 1.0 | 100% of A colocalizes with B |
| 0.5 | 50% overlap |
| 0 | No overlap |

```python
m1 = res['results']['total_image']['manders_m1']  # A overlapping B
m2 = res['results']['total_image']['manders_m2']  # B overlapping A
```

### Jaccard Index

Measures overlap of thresholded (binary) regions:

$$J = \frac{|A \cap B|}{|A \cup B|}$$

```python
jaccard = res['results']['total_image']['jaccard']
```

---

## compute_colocalization() Parameters

```python
res = cr.compute_colocalization(
    # Required
    image=corrected_stack,        # 4D array (Z, Y, X, C) or file path
    mask=mask_path,               # Segmentation mask (2D or path)
    channel_a="ALIX",             # First channel name or index
    channel_b="LAMP1",            # Second channel name or index
    
    # Channel identification
    channel_names=["DAPI", "ALIX", "Phalloidin", "LAMP1"],
    
    # Thresholding
    thresholding="otsu",          # "none", "otsu", "costes", "fixed"
    fixed_thresholds=(10, 15),    # For thresholding="fixed"
    min_threshold_sigma=3.0,      # Signal floor: threshold >= mean + N*std
    
    # Cell filtering
    min_area="auto",              # Minimum cell area (pixels or "auto")
    min_area_fraction=0.9,        # For "auto": fraction of median area
    max_border_fraction=0.2,      # Remove cells touching border
    border_margin_px=1,           # Border margin in pixels
    drop_label_1=True,            # Remove label 1 (Cellpose background)
    
    # Advanced options
    pearson_winsor_clip=0.1,      # Winsorize outliers for Pearson
    manders_weighting="voxel",    # "voxel" (global) or "slice" (per-z)
    
    # Visualization
    plot_mask=True,               # Generate mask visualization
    plot_mask_save="mask_filtered.png",  # Save path
    
    # Output
    output_json="results.json",   # Save results to JSON
)
```

---

## Thresholding Methods

### None (Raw Values)

Use raw intensity values without thresholding:

```python
res = cr.compute_colocalization(
    ...,
    thresholding="none",
)
```

### Otsu

Automatic threshold using Otsu's method:

```python
res = cr.compute_colocalization(
    ...,
    thresholding="otsu",
)
```

### Costes

Automatic threshold using Costes' method (iterative):

```python
res = cr.compute_colocalization(
    ...,
    thresholding="costes",
)
```

### Fixed

Manual threshold values:

```python
res = cr.compute_colocalization(
    ...,
    thresholding="fixed",
    fixed_thresholds=(10.0, 15.0),  # (threshold_A, threshold_B)
)
```

---

## Signal Quality Control

### The Problem with Noise-Only Channels

Automatic thresholding (Otsu, Costes) assumes both channels have real signal. When a channel contains only noise (e.g., a negative control), these methods set thresholds near the noise floor, causing inflated Manders/Jaccard values.

### Signal Floor (`min_threshold_sigma`)

The `min_threshold_sigma` parameter ensures thresholds are meaningfully above the noise floor:

$$\text{threshold}_{floor} = \mu + \sigma \times \text{min\_threshold\_sigma}$$

If the computed threshold falls below this floor, it's automatically raised:

```python
res = cr.compute_colocalization(
    ...,
    thresholding="otsu",           # or "costes"
    min_threshold_sigma=3.0,       # Threshold must be >= mean + 3*std
)
```

| Value | Effect |
|-------|--------|
| 0 | Disabled (default) |
| 2.0 | Conservative - allows some noise |
| 3.0 | Recommended for most cases |
| 5.0 | Aggressive - excludes more signal |

### Detecting Floored Thresholds

Results include information about whether thresholds were raised to the floor:

```python
for z_info in res['results']['total_image']['thresholds_per_z']:
    if z_info['a_floored']:
        print(f"Z={z_info['z']}: Channel A threshold raised from {z_info['a_original']:.2f} to {z_info['t_a']:.2f}")
```

### Example: Negative Control Analysis

```python
# Negative control where ALIX channel has no real signal
res = cr.compute_colocalization(
    image=corrected_stack,
    mask=mask_path,
    channel_a="ALIX",              # No signal (noise only)
    channel_b="LAMP1",
    channel_names=channel_names,
    thresholding="otsu",
    min_threshold_sigma=3.0,       # Ensures noise isn't counted as signal
)

# Without min_threshold_sigma: Manders ~0.15-0.25 (inflated!)
# With min_threshold_sigma=3.0: Manders ~0.0 (correct for negative control)
```

---

## Cell Filtering

### Automatic Area Filtering

Remove cells below a fraction of median area:

```python
res = cr.compute_colocalization(
    ...,
    min_area="auto",
    min_area_fraction=0.9,  # Keep cells >= 90% of median area
)
```

### Fixed Area Threshold

```python
res = cr.compute_colocalization(
    ...,
    min_area=5000,  # Minimum 5000 pixels
)
```

### Border Cell Removal

Remove cells touching the image border:

```python
res = cr.compute_colocalization(
    ...,
    max_border_fraction=0.2,   # Remove if >20% touches border
    border_margin_px=1,        # Border margin
)
```

---

## Results Structure

```python
res = cr.compute_colocalization(...)

# Per-cell results
per_label = res['results']['per_label']
# List of dicts, one per cell:
# {
#     'label': int,
#     'pearson_r': float,
#     'manders_m1': float,
#     'manders_m2': float,
#     'jaccard': float,
#     'n_voxels': int,
#     ...
# }

# Total image results
total = res['results']['total_image']
# {
#     'pearson_r': float,
#     'manders_m1': float,
#     'manders_m2': float,
#     'jaccard': float,
#     'n_voxels': int,
#     ...
# }

# Filter information
filter_info = res['filter_info']
# {
#     'kept_labels': [...],
#     'removed_labels': [...],
#     'removal_reasons': {...},
# }
```

---

## Working with Results

### Convert to DataFrame

```python
import pandas as pd

# Per-cell metrics
df_cells = pd.DataFrame(res['results']['per_label'])
df_cells = df_cells.sort_values('label')

# Summary statistics
print(f"Mean Pearson: {df_cells['pearson_r'].mean():.3f}")
print(f"Mean M1: {df_cells['manders_m1'].mean():.3f}")

# Total image (single row)
df_total = pd.DataFrame([res['results']['total_image']])
```

### Visualize Filtered Cells

```python
from IPython.display import Image, display

res = cr.compute_colocalization(
    ...,
    plot_mask=True,
    plot_mask_save="mask_filtered.png",
)

# Display in notebook
display(Image(filename="mask_filtered.png"))
```

### Save to JSON

```python
res = cr.compute_colocalization(
    ...,
    output_json="colocalization_results.json",
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

# 5. Colocalization
corrected_stack = np.stack([results[ch][0] for ch in channel_names], axis=-1)

coloc = cr.compute_colocalization(
    image=corrected_stack,
    mask=seg.mask_path,
    channel_a="ALIX",
    channel_b="LAMP1",
    channel_names=channel_names,
    thresholding="otsu",
    min_threshold_sigma=3.0,      # Prevents noise from inflating metrics
    min_area="auto",
    min_area_fraction=0.9,
    plot_mask=True,
    plot_mask_save="./output/mask_filtered.png",
)

# 6. Analyze results
df_cells = pd.DataFrame(coloc['results']['per_label'])
total = coloc['results']['total_image']

print(f"Analyzed {len(df_cells)} cells")
print(f"Total Pearson r: {total['pearson_r']:.3f}")
print(f"Total Manders M1: {total['manders_m1']:.3f}")
print(f"Total Manders M2: {total['manders_m2']:.3f}")
print(f"Total Jaccard: {total['jaccard']:.3f}")

# Save per-cell results
df_cells.to_csv("./output/per_cell_colocalization.csv", index=False)
```

---

## Interpreting Results

### Pearson Correlation

| Range | Interpretation |
|-------|----------------|
| 0.8 - 1.0 | Strong positive correlation |
| 0.5 - 0.8 | Moderate positive correlation |
| 0.2 - 0.5 | Weak positive correlation |
| -0.2 - 0.2 | No correlation |
| < -0.2 | Negative correlation (anti-colocalization) |

### Manders Coefficients

| M1 / M2 | Interpretation |
|---------|----------------|
| > 0.9 | Almost complete overlap |
| 0.6 - 0.9 | Substantial overlap |
| 0.3 - 0.6 | Partial overlap |
| < 0.3 | Minimal overlap |

### Common Patterns

| Pattern | Pearson | M1 | M2 |
|---------|---------|----|----|
| Perfect colocalization | ~1.0 | ~1.0 | ~1.0 |
| A contains B | High | Low | High |
| B contains A | High | High | Low |
| Partial overlap | Moderate | Moderate | Moderate |
| No colocalization | ~0 | ~0 | ~0 |
| Negative control (no signal in A) | ~0 | ~0 | ~0 |

> **Note**: Without `min_threshold_sigma`, negative controls can show artificially inflated Manders values (0.15-0.30) because automatic thresholding sets thresholds near the noise floor. Use `min_threshold_sigma=3.0` to ensure accurate results for negative controls.

