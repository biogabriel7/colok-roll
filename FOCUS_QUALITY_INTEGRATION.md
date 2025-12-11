# Focus Quality Metrics Integration - Complete

This document summarizes the integration of Piao et al. (2025) focus quality metrics into the Colokroll library.

## What Was Done

The focus measure quality metrics from `focus_measure_quality.py` have been fully integrated into the z-slice selection workflow. Users can now objectively evaluate and compare focus methods based on quantitative morphological characteristics of focus curves.

## Changes Made

### 1. Updated `ZSliceSelectionResult` Dataclass

Added optional `quality_metrics` field:

```python
@dataclass(frozen=True)
class ZSliceSelectionResult:
    # ... existing fields ...
    quality_metrics: Optional[FocusMeasureQuality] = None
```

### 2. Enhanced `select_z_slices()` Function

Added parameters to compute quality metrics:

- `compute_quality: bool = False` - Enable quality metrics computation
- `step_distance: float = 1.0` - Physical z-step distance in µm
- `n_fitting_points: int = 5` - Number of points for linear fitting

### 3. New `benchmark_focus_methods()` Function

Compare multiple focus methods with quality metrics:

```python
def benchmark_focus_methods(
    img: np.ndarray,
    methods: Optional[list] = None,
    step_distance: float = 1.0,
    **kwargs
) -> Dict[str, ZSliceSelectionResult]
```

### 4. Enhanced `compare_strategies()` Function

Added optional quality metrics computation for strategy comparisons:

- `compute_quality: bool = False`
- `step_distance: float = 1.0`
- `n_fitting_points: int = 5`

### 5. Updated `StrategyComparisonResult` Dataclass

Added optional quality comparison field:

```python
quality_comparison: Optional[Dict[str, FocusMeasureQuality]] = None
```

### 6. Updated Module Exports

Added to `colokroll.imaging_preprocessing.__init__.py`:

- `benchmark_focus_methods`
- `FocusMeasureQuality`
- `CurveSegmentation`
- `compute_focus_measure_quality`
- `extend_z_slice_result_with_quality`
- `plot_focus_curve_analysis`

## Usage Examples

### Basic Usage with Quality Metrics

```python
from colokroll.imaging_preprocessing import select_z_slices

result = select_z_slices(
    z_stack,
    method="combined",
    compute_quality=True,
    step_distance=0.5,  # µm between slices
)

# Access quality metrics
q = result.quality_metrics
print(f"Steep width (Ws): {q.Ws:.2f} µm")
print(f"Steep/gradual ratio (Rsg): {q.Rsg:.3f}")
print(f"Peak curvature (Cp): {q.Cp:.4f}")
```

### Compare Focus Methods

```python
from colokroll.imaging_preprocessing import benchmark_focus_methods

results = benchmark_focus_methods(
    z_stack,
    methods=["laplacian", "tenengrad", "combined", "fft"],
    step_distance=0.5,
)

# Print comparison
for method, result in results.items():
    q = result.quality_metrics
    print(f"{method}: Ws={q.Ws:.2f}, Rsg={q.Rsg:.3f}")

# Select best method by Rsg
best = max(results.items(), key=lambda x: x[1].quality_metrics.Rsg)
print(f"Best method: {best[0]}")
```

### Strategy Comparison with Quality

```python
from colokroll.imaging_preprocessing import compare_strategies

strategies = [
    {"name": "FFT Auto", "method": "fft", "strategy": "closest_to_peak"},
    {"name": "Combined k=15", "method": "combined", "keep_top": 15},
]

comparison = compare_strategies(
    z_stack,
    strategies=strategies,
    compute_quality=True,
    step_distance=0.5,
)

# Access quality comparison
for name, quality in comparison.quality_comparison.items():
    print(f"{name}: Rsg={quality.Rsg:.3f}")
```

### Visualize Focus Curve

```python
from colokroll.imaging_preprocessing import (
    select_z_slices,
    plot_focus_curve_analysis,
)

result = select_z_slices(
    z_stack,
    compute_quality=True,
    step_distance=0.5,
)

plot_focus_curve_analysis(
    result.scores_agg,
    result.quality_metrics,
    output_path="focus_analysis.png",
)
```

## Quality Metrics Reference

### Ws (Steep Width)
- **Lower is better**
- Narrow steep region → High sensitivity to focus changes
- Typical good value: < 5.0 µm

### Rsg (Steep-to-Gradual Ratio)
- **Higher is better**
- High ratio → Better discrimination of in-focus vs. out-of-focus
- Typical good value: > 2.0

### Cp (Peak Curvature)
- **Higher is better**
- High curvature → More sensitive to small focal deviations
- Typical good value: > 0.01

### FWHM (Full Width at Half Maximum)
- **Lower is better**
- Traditional sharpness metric
- Compare with Ws for consistency

### Additional Metrics
- **Sp**: Steepness (higher = better)
- **RRMSE**: Noise robustness (lower = better, requires noisy scores)
- **is_unimodal**: Curve has single peak (should be True)

## Workflow Recommendations

### 1. Initial Method Selection

For new tissue types or imaging protocols:

1. Collect 3-5 representative z-stacks
2. Run `benchmark_focus_methods()` on each
3. Average quality metrics across samples
4. Select method with highest average Rsg or lowest Ws
5. Use selected method for all subsequent analysis

### 2. Quality Control

Add quality checks to your pipeline:

```python
result = select_z_slices(z_stack, compute_quality=True)

if not result.quality_metrics.is_unimodal:
    print("Warning: Non-unimodal focus curve detected")
    
if result.quality_metrics.Rsg < 1.5:
    print("Warning: Poor focus discrimination (low Rsg)")
```

### 3. Method Comparison Studies

Document which methods work best for specific tissue types:

```python
# Benchmark on multiple samples
all_results = []
for sample in samples:
    results = benchmark_focus_methods(sample, step_distance=0.5)
    all_results.append(results)

# Compute average metrics per method
# Select best method based on data
```

## Important Notes

1. **Unimodality**: Piao et al. metrics assume unimodal curves. Always check `is_unimodal`.

2. **Step Distance**: Provide accurate z-step distance from microscope metadata for correct physical units.

3. **Fitting Points**: Use 3 points for small stacks (<20 slices), 5-7 for larger stacks.

4. **Tissue-Specific**: Different tissues may favor different methods. Benchmark on representative samples.

5. **Reproducibility**: Quality metrics enable objective, reproducible method selection decisions.

## Demo Script

A comprehensive demonstration is available:

```bash
python examples/focus_quality_metrics_demo.py
```

This demo shows:
- Basic usage with quality metrics
- Method benchmarking
- Strategy comparison
- Visualization
- Tissue-specific selection workflow

## References

Piao, W., Han, Y., Hu, L., & Wang, C. (2025). Quantitative Evaluation of Focus Measure Operators in Optical Microscopy. *Sensors*, 25, 3144.

## Files Modified

- `colokroll/imaging_preprocessing/z_slice_detection.py` - Core integration
- `colokroll/imaging_preprocessing/__init__.py` - Export new functionality
- `examples/focus_quality_metrics_demo.py` - Comprehensive demo (new)
- `FOCUS_QUALITY_INTEGRATION.md` - This document (new)

## Testing

All changes pass linting with no errors. The integration maintains backward compatibility - existing code will continue to work without changes.

To enable quality metrics, simply add `compute_quality=True` to function calls.

