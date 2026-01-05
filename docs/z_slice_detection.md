# Z-Slice Detection

Automatically identify and filter out-of-focus slices from confocal z-stacks using focus quality metrics.

---

## Overview

Confocal z-stacks often contain slices that are out of focus (above or below the sample). ColokRoll provides tools to:

1. **Compute focus scores** for each z-slice using various metrics
2. **Compare strategies** to find the best approach for your data
3. **Filter slices** to keep only in-focus regions

---

## Quick Start

```python
import colokroll as cr

# Load image
loader = cr.ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")

# Simple filtering with defaults
result = cr.select_z_slices(
    image,
    method="combined",
    strategy="relative",
    threshold=0.6,
)

# Apply filtering
filtered_image = image[result.indices_keep]
print(f"Kept {len(result.indices_keep)} / {image.shape[0]} slices")
```

---

## Focus Metrics

### Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `laplacian` | Laplacian variance | General purpose |
| `tenengrad` | Sobel gradient magnitude | Edge-rich samples |
| `fft` | High-frequency FFT content | Fine structures |
| `combined` | Weighted combination of Laplacian + Tenengrad | Most reliable |

### How Metrics Work

Each metric computes a score per z-slice per channel:

```python
# Compute focus scores directly
from colokroll.imaging_preprocessing import compute_focus_scores

scores_zc, scores_agg = compute_focus_scores(
    image,
    method="combined",
    normalize=True,           # Normalize per channel
    clip_percent=1.0,         # Clip outliers
)

# scores_zc: shape (Z, C) - per-slice, per-channel
# scores_agg: shape (Z,) - aggregated across channels
```

---

## Detection Strategies

### Strategy Comparison

Different strategies for deciding which slices to keep:

| Strategy | Description | Use When |
|----------|-------------|----------|
| `relative` | Keep slices above threshold × peak score | Variable focus quality |
| `closest` | Keep k slices closest to peak | Fixed number needed |
| `topk` | Keep top k scoring slices | Fixed number, any position |
| `auto` | Automatic selection based on score distribution | General purpose |

### Compare All Strategies

```python
# Compare multiple strategies visually
comparison = cr.compare_strategies(
    image,
    save_plots=True,         # Save decision matrix
    display_inline=True,     # Show in notebook
    compute_quality=True,    # Compute quality metrics
)

# View what's available
print(comparison.strategy_names)
# ['FFT + Closest (Auto 0.8)', 'FFT + Closest (Auto 0.7)', 
#  'FFT + Closest (k=14)', 'Combined + Closest (Auto)', ...]

# Access results for a specific strategy
result = comparison.results["FFT + Closest (k=14)"]
print(f"Kept slices: {result.indices_keep}")
print(f"Removed slices: {result.indices_remove}")
```

### Output Files

When `save_plots=True`, generates:

```
strategy_comparison/
├── decision_matrix_heatmap.png  # Which strategy keeps which slices
├── z_slice_gallery.png          # Visual preview of all slices
└── comparison_summary.txt       # Statistics for each strategy
```

---

## Using select_z_slices()

### Parameters

```python
result = cr.select_z_slices(
    image,                        # Input: (Z, Y, X, C) array
    axes=None,                    # Optional axis order string
    
    # Focus metric
    method="combined",            # "laplacian", "tenengrad", "fft", "combined"
    aggregation="median",         # How to combine channels: "median", "mean", "max"
    
    # Strategy
    strategy="relative",          # "relative", "closest", "topk", "auto"
    threshold=0.6,                # For "relative": fraction of peak
    keep_top=None,                # For "closest"/"topk": number of slices
    auto_keep_fraction=0.8,       # For "auto": fraction of peak
    
    # Preprocessing
    smooth=3,                     # Smoothing window for scores
    normalize=True,               # Normalize scores per channel
    clip_percent=1.0,             # Percentile clipping
    
    # Quality metrics (optional)
    compute_quality=False,        # Compute Piao et al. metrics
    step_distance=1.0,            # Z-step in micrometers
)
```

### Return Value

```python
result = cr.select_z_slices(image, ...)

# Access results
result.indices_keep      # np.ndarray: Indices of kept slices
result.indices_remove    # np.ndarray: Indices of removed slices
result.scores_agg        # np.ndarray: Aggregated focus scores
result.smoothed_scores   # np.ndarray: Smoothed scores
result.method            # str: Method used
result.strategy          # str: Strategy used
result.quality_metrics   # Optional: Piao et al. quality metrics
```

---

## Strategy Details

### Relative Strategy

Keep all slices with scores above `threshold × peak_score`:

```python
result = cr.select_z_slices(
    image,
    strategy="relative",
    threshold=0.6,    # Keep slices with score >= 60% of peak
)
```

### Closest Strategy

Keep the k slices closest to the peak (contiguous region around peak):

```python
result = cr.select_z_slices(
    image,
    strategy="closest",
    keep_top=14,      # Keep 14 slices centered on peak
)

# Or auto-determine k based on score threshold
result = cr.select_z_slices(
    image,
    strategy="closest",
    auto_keep_fraction=0.8,  # Auto-select k
)
```

### TopK Strategy

Keep the top k scoring slices (not necessarily contiguous):

```python
result = cr.select_z_slices(
    image,
    strategy="topk",
    keep_top=20,      # Keep 20 highest-scoring slices
)
```

---

## Quality Metrics (Piao et al.)

Objective metrics for evaluating focus curve quality:

```python
result = cr.select_z_slices(
    image,
    compute_quality=True,
    step_distance=0.5,        # Z-step in micrometers
    n_fitting_points=5,       # Points for linear fitting
)

# Access quality metrics
qm = result.quality_metrics
print(f"Rsg (steep-to-gradual ratio): {qm.Rsg:.2f}")  # Higher = better
print(f"Ws (steep width): {qm.Ws:.2f}")               # Lower = better
print(f"Cp (peak curvature): {qm.Cp:.2f}")            # Higher = better
print(f"Is unimodal: {qm.is_unimodal}")
```

### Auto-Select Best Method

Automatically benchmark and select the best focus method:

```python
result = cr.auto_select_best_method(
    image,
    step_distance=0.5,
    ranking_metric="Rsg",     # "Rsg", "Ws", "Cp", or "composite"
    require_unimodal=True,    # Only consider unimodal curves
    verbose=True,
)

print(f"Auto-selected method: {result.method}")
```

---

## Examples

### Basic Filtering

```python
# Simple relative threshold
result = cr.select_z_slices(image, threshold=0.6)
filtered = image[result.indices_keep]
```

### Fixed Number of Slices

```python
# Always keep exactly 14 slices
result = cr.select_z_slices(image, strategy="closest", keep_top=14)
```

### Compare and Choose

```python
# Compare strategies, then apply chosen one
comparison = cr.compare_strategies(image, display_inline=True)
result = comparison.results["FFT + Closest (k=14)"]
filtered = image[result.indices_keep]
```

### Batch Processing

```python
# Use consistent parameters across all images
def filter_zslices(image):
    result = cr.select_z_slices(
        image,
        method="fft",
        strategy="closest",
        keep_top=14,
    )
    return image[result.indices_keep]
```

