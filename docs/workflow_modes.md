# Workflow Modes

ColokRoll supports two workflow modes for processing microscopy data: **Exploratory Mode** for parameter calibration and **Batch Mode** for production processing.

---

## Exploratory Mode (Parameter Calibration)

Use exploratory mode on your first image (or a representative sample) to visually inspect results and select optimal parameters for your dataset.

### When to Use

- Processing a new dataset for the first time
- Calibrating parameters with negative control samples
- Comparing different strategies before batch processing
- Debugging or optimizing results

### Z-Slice Detection Calibration

Compare multiple strategies to find the best one for your data:

```python
import colokroll as cr

# Load your image
loader = cr.ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")
loader.rename_channels(['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'])

# Compare all available strategies
comparison = cr.compare_strategies(
    image,
    save_plots=True,           # Save decision matrix and gallery
    compute_quality=False,     # Skip quality metrics for speed
    display_inline=True,       # Show plots in notebook
)

# View available strategies
print(comparison.strategy_names)
# ['FFT + Closest (Auto 0.8)', 'FFT + Closest (k=14)', ...]

# Visually inspect the decision matrix and slice gallery
# Then pick the strategy that keeps the slices you want
result = comparison.results["FFT + Closest (k=14)"]

# Apply the filtering
filtered_image = image[result.indices_keep]
print(f"Kept {len(result.indices_keep)} / {image.shape[0]} slices")
```

### Background Subtraction with Negative Control

Use a negative control sample to calibrate background subtraction parameters:

```python
bg_subtractor = cr.BackgroundSubtractor()

# Process channels - mark known negative controls
results = {}
for i, ch in enumerate(loader.get_channel_names()):
    corrected, meta = bg_subtractor.subtract_background(
        image=filtered_image[:, :, :, i],
        channel_name=ch,
        is_negative_control=(ch == "ALIX"),  # ALIX is our negative control
    )
    results[ch] = (corrected, meta)

# Check the negative control validation metrics
alix_meta = results["ALIX"][1]
print("Negative control validation:")
print(alix_meta['negative_control_validation'])
# {'residual_mean': 1.7, 'residual_std': 2.6, 'zero_fraction': 0.85, ...}

# Extract the optimized parameters for use on positive samples
best_method = alix_meta['method']
best_params = alix_meta['parameters_used']
print(f"Best method: {best_method}")
print(f"Best params: {best_params}")
```

### Visualize Results

```python
# Plot background subtraction comparison
fig = bg_subtractor.plot_background_subtraction_comparison(
    original_data=filtered_image,
    corrected_results=results,
    channel_names=loader.get_channel_names(),
    z_slice=filtered_image.shape[0] // 2,
)
```

---

## Batch Mode (Production Processing)

Use batch mode to apply validated parameters consistently across all images in your dataset.

### When to Use

- Processing multiple images with the same acquisition settings
- Running automated pipelines
- Reproducible analysis with fixed parameters

### Apply Calibrated Parameters

```python
import colokroll as cr
from pathlib import Path

# Parameters from exploratory mode
ZSLICE_STRATEGY = "closest"
ZSLICE_METHOD = "fft"
ZSLICE_KEEP_TOP = 14

ALIX_BG_METHOD = "two_stage"
ALIX_BG_PARAMS = {'sigma_stage1': 0.5, 'radius_stage2': 5, 'light_background': False}

def process_image(image_path: Path) -> dict:
    """Process a single image with calibrated parameters."""
    
    # 1. Load
    loader = cr.ImageLoader()
    image = loader.load_image(image_path)
    loader.rename_channels(['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'])
    channel_names = loader.get_channel_names()
    
    # 2. Z-slice selection (fixed parameters)
    result = cr.select_z_slices(
        image,
        method=ZSLICE_METHOD,
        strategy=ZSLICE_STRATEGY,
        keep_top=ZSLICE_KEEP_TOP,
    )
    filtered_image = image[result.indices_keep]
    
    # 3. Background subtraction
    bg_subtractor = cr.BackgroundSubtractor()
    results = {}
    
    for i, ch in enumerate(channel_names):
        if ch == "ALIX":
            # Use calibrated parameters from negative control
            corrected, meta = bg_subtractor.subtract_background(
                image=filtered_image[:, :, :, i],
                method=ALIX_BG_METHOD,
                channel_name=ch,
                **ALIX_BG_PARAMS,
            )
        else:
            # Auto-select for other channels
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
        save_basename=image_path.stem,
    )
    
    # 5. Colocalization
    import numpy as np
    corrected_stack = np.stack([results[ch][0] for ch in channel_names], axis=-1)
    
    coloc = cr.compute_colocalization(
        image=corrected_stack,
        mask=seg.mask_path,
        channel_a="ALIX",
        channel_b="LAMP1",
        channel_names=channel_names,
        thresholding="otsu",
    )
    
    return {
        'image_path': str(image_path),
        'n_cells': len(coloc['results']['per_label']),
        'pearson': coloc['results']['total_image']['pearson_r'],
        'manders_m1': coloc['results']['total_image']['manders_m1'],
        'manders_m2': coloc['results']['total_image']['manders_m2'],
    }

# Process all images
image_paths = list(Path("./data").glob("*.ome.tiff"))
results = [process_image(p) for p in image_paths]
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPLORATORY MODE                            │
│  (First image / Negative control)                               │
├─────────────────────────────────────────────────────────────────┤
│  1. compare_strategies() → Pick best z-slice strategy           │
│  2. subtract_background(is_negative_control=True) → Get params  │
│  3. Visualize and validate results                              │
│  4. Extract parameters for batch mode                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BATCH MODE                                │
│  (All remaining images)                                         │
├─────────────────────────────────────────────────────────────────┤
│  1. select_z_slices() with fixed strategy/params                │
│  2. subtract_background() with calibrated params                │
│  3. Segmentation and colocalization                             │
│  4. Aggregate results across all images                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Best Practices

1. **Use representative samples** for calibration - pick images that represent the typical quality and signal levels in your dataset

2. **Validate negative controls** - check that `residual_mean` and `residual_std` are low after background subtraction

3. **Save calibration parameters** - store the validated parameters (method, sigma, radius, etc.) for reproducibility

4. **Document your choices** - record why you selected specific strategies and parameters

5. **Re-calibrate when needed** - if acquisition settings change, re-run exploratory mode

