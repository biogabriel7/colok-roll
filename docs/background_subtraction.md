# Background subtraction

Remove background fluorescence from confocal microscopy images with GPU acceleration and negative control support.

---

## Overview

Background subtraction removes diffuse background fluorescence to improve signal-to-noise ratio. ColokRoll provides:

- Multiple methods: rolling ball, Gaussian, morphological, two-stage
- GPU acceleration: CUDA (CuPy) and MPS (Apple Silicon) backends
- Automatic parameter selection: grid search with scoring
- Negative control support: calibrate parameters using negative control samples

---

## Quick start

```python
import colokroll as cr

# Initialize with GPU acceleration
bg_subtractor = cr.BackgroundSubtractor()

# Auto-select best method and parameters
corrected, meta = bg_subtractor.subtract_background(
    image=channel_data,       # 3D array: (Z, Y, X)
    channel_name="LAMP1",     # For channel-specific defaults
)
```

---

## Methods

### Available methods

| Method | Description | Best for |
|--------|-------------|----------|
| `rolling_ball` | Rolling ball algorithm | Uneven illumination |
| `gaussian` | 3D Gaussian subtraction | Diffuse haze |
| `morphological` | Morphological opening | Structured backgrounds |
| `two_stage` | Gaussian + Rolling ball | Combined benefits |

### Using specific methods

```python
# Rolling ball
corrected, meta = bg_subtractor.subtract_background(
    image=data,
    method="rolling_ball",
    radius=50,                # Ball radius in pixels
    light_background=False,   # Dark background (typical for fluorescence)
)

# Gaussian
corrected, meta = bg_subtractor.subtract_background(
    image=data,
    method="gaussian",
    sigma=14,                 # Gaussian sigma
)

# Two-stage (Gaussian then Rolling ball)
corrected, meta = bg_subtractor.subtract_background(
    image=data,
    method="two_stage",
    sigma_stage1=14,          # First stage: Gaussian sigma
    radius_stage2=30,         # Second stage: Rolling ball radius
)
```

---

## Automatic parameter selection

When `method` is not specified (or set to `"auto"`), ColokRoll automatically searches for optimal parameters:

```python
# Auto mode (default)
corrected, meta = bg_subtractor.subtract_background(
    image=data,
    channel_name="LAMP1",     # Uses channel-specific search grids
)

# Check what was selected
print(f"Method: {meta['method']}")
print(f"Parameters: {meta['parameters_used']}")
print(f"Candidates tested: {meta['auto_candidates_tested']}")
```

### Channel-specific defaults

The auto-search uses channel-specific parameter grids:

| Channel | Default method | Typical parameters |
|---------|---------------|-------------------|
| `LAMP1` | two_stage | sigma=14, radius=30 |
| `Phalloidin` | rolling_ball | radius=50-70 |
| `ALIX` | gaussian | sigma=10-16 |
| `DAPI` | rolling_ball | radius=100-140 |

---

## Negative control support

Use negative control samples to calibrate background subtraction parameters.

### What is a negative control?

A negative control is a sample where you know a specific channel should have no signal (e.g., ALIX channel in a sample without ALIX staining). This lets you:

1. Validate that background subtraction is working correctly
2. Calibrate parameters that minimize residual signal
3. Transfer validated parameters to positive samples

### Using negative control mode

```python
# Process a negative control channel
corrected, meta = bg_subtractor.subtract_background(
    image=alix_data,
    channel_name="ALIX",
    is_negative_control=True,  # Enable negative control scoring
)

# Check validation metrics
validation = meta['negative_control_validation']
print(f"Residual mean: {validation['residual_mean']:.2f}")
print(f"Residual std: {validation['residual_std']:.2f}")
print(f"Zero fraction: {validation['zero_fraction']:.1%}")
print(f"95th percentile: {validation['residual_percentile_95']:.2f}")
```

### How negative control scoring works

Standard scoring optimizes for:
- Background reduction
- Contrast preservation
- Gradient preservation
- SSIM similarity

Negative control scoring optimizes for:
- Mean reduction: lower residual mean intensity
- Std reduction: lower residual standard deviation
- Zero fraction: more pixels near zero

The algorithm balances these to prevent complete signal flattening while minimizing residual signal.

### Transferring parameters to positive samples

```python
# 1. Calibrate on negative control
corrected_neg, meta_neg = bg_subtractor.subtract_background(
    image=negative_alix_data,
    channel_name="ALIX",
    is_negative_control=True,
)

# 2. Extract validated parameters
best_method_raw = meta_neg['method']
best_params = meta_neg['parameters_used']

# Map display name to method name
if 'gaussian+rolling_ball' in best_method_raw:
    best_method = "two_stage"
elif 'gaussian' in best_method_raw:
    best_method = "gaussian"
elif 'rolling_ball' in best_method_raw:
    best_method = "rolling_ball"

print(f"Validated method: {best_method}")
print(f"Validated params: {best_params}")

# 3. Apply to positive samples
corrected_pos, meta_pos = bg_subtractor.subtract_background(
    image=positive_alix_data,
    method=best_method,
    channel_name="ALIX",
    **best_params,  # Use calibrated parameters
)
```

---

## Processing multiple channels

```python
bg_subtractor = cr.BackgroundSubtractor()
channel_names = ['DAPI', 'ALIX', 'Phalloidin', 'LAMP1']
results = {}

for i, ch in enumerate(channel_names):
    ch_data = image[:, :, :, i]

    corrected, meta = bg_subtractor.subtract_background(
        image=ch_data,
        channel_name=ch,
        is_negative_control=(ch == "ALIX"),  # Mark negative controls
    )

    results[ch] = (corrected, meta)

# Visualize results
fig = bg_subtractor.plot_background_subtraction_comparison(
    original_data=image,
    corrected_results=results,
    channel_names=channel_names,
    z_slice=image.shape[0] // 2,
)
```

---

## GPU acceleration

### Backends

ColokRoll automatically selects the best available backend:

| Backend | Requirements | Speed |
|---------|--------------|-------|
| CUDA | CuPy + NVIDIA GPU | 10-50x faster |
| MPS | PyTorch + Apple Silicon | 5-20x faster |
| CPU | NumPy/SciPy | Baseline |

### Installation

```bash
# CUDA (NVIDIA GPUs)
pip install cupy-cuda12x  # For CUDA 12.x

# MPS (Apple Silicon)
pip install torch kornia
```

### Checking backend

```python
bg_subtractor = cr.BackgroundSubtractor()
print(f"Using backend: {bg_subtractor.backend}")
# "cuda", "mps", or "cpu"
```

---

## Visualization

### Comparison plot

```python
fig = bg_subtractor.plot_background_subtraction_comparison(
    original_data=image,           # Original 4D array
    corrected_results=results,     # Dict of (corrected, meta) tuples
    channel_names=channel_names,
    z_slice=12,                    # Slice to display
    figsize=(20, 12),
)
fig.savefig("bg_comparison.png", dpi=200)
```

Generates a 3-row plot for each channel:
1. Original image with colorbar
2. Corrected image with method info
3. Intensity histogram comparing original vs corrected

---

## Configuration

### BackgroundSubtractionConfig

```python
from colokroll.core.config import BackgroundSubtractionConfig

config = BackgroundSubtractionConfig(
    default_method="rolling_ball",
    default_radius=50,
    default_sigma=10,
    clip_negative_values=True,    # Clip negative values to 0
    normalize_output=False,        # Don't normalize output
)

bg_subtractor = cr.BackgroundSubtractor(config=config)
```

### Auto-search weights

Customize scoring weights for auto-selection:

```python
bg_subtractor = cr.BackgroundSubtractor()
bg_subtractor.auto_weights = (
    0.7,   # w_bg: Background reduction
    0.4,   # w_contrast: Contrast preservation
    0.2,   # w_grad: Gradient preservation
    0.1,   # w_zero: Zero fraction penalty
)
```

---

## Metadata

The returned metadata includes:

```python
corrected, meta = bg_subtractor.subtract_background(...)

# Key metadata fields
meta['method']              # Method used (e.g., "gaussian_cuda")
meta['parameters_used']     # Dict of parameters
meta['original_shape']      # Input shape
meta['gpu_accelerated']     # True if GPU was used
meta['is_negative_control'] # True if negative control mode

# Auto-selection info (when method="auto")
meta['auto_selected']       # True
meta['auto_candidates_tested']  # Number of parameter combinations tested
meta['auto_method_scores']  # Scores for each method
meta['auto_top3_methods']   # Top 3 methods with scores

# Negative control validation (when is_negative_control=True)
meta['negative_control_validation'] = {
    'residual_mean': float,
    'residual_std': float,
    'residual_percentile_95': float,
    'residual_percentile_99': float,
    'zero_fraction': float,
}
```
