# Cell Segmentation with Cellpose

This module provides two interfaces for cell segmentation using the Cellpose Gradio Space.

## Quick Start (Recommended)

For most users, use the simple interface:

```python
from colokroll.io import ImageLoader
from colokroll.analysis.segmentation import segment_from_loader

# Load image
loader = ImageLoader()
image = loader.load_image("data.ome.tiff")

# Segment (one line!)
mask_path, outlines_path, mask = segment_from_loader(
    loader, "Phalloidin", "DAPI",
    output_dir="results"
)

print(f"Found {mask.max()} cells")
```

## Two Interfaces

### 1. Simple Interface (Recommended) ✨

**Functions**: `segment_cells()`, `segment_from_loader()`

**Use when**:
- You want it to just work
- You're following the proven notebook workflow
- You need reliable, reproducible results

**Features**:
- ✅ Uses the exact workflow tested to work reliably
- ✅ Simple, clear parameters
- ✅ Works with MIP composites (recommended)
- ✅ Automatic retry with fallback resize values
- ✅ Built-in error handling

**Example**:
```python
from colokroll.analysis.segmentation import segment_cells
from colokroll.io import ImageLoader, MIPCreator

# Create MIPs
loader = ImageLoader()
image = loader.load_image("data.ome.tiff")

phall_stack = loader.extract_channel(image, "Phalloidin")
dapi_stack = loader.extract_channel(image, "DAPI")

mip_creator = MIPCreator()
phall_mip = mip_creator.create_mip(phall_stack, method="max")
dapi_mip = mip_creator.create_mip(dapi_stack, method="max")

# Segment
mask_path, outlines_path, mask = segment_cells(
    phall_mip, dapi_mip,
    output_dir="results",
    filename_stem="sample1"
)
```

### 2. Advanced Interface (CellSegmenter Class)

**Class**: `CellSegmenter`

**Use when**:
- You need fine-grained control over segmentation parameters
- You want to use middle z-slices instead of MIPs
- You need to work with preprocessing results dictionaries
- You're building custom pipelines

**Features**:
- 🔧 Multiple projection methods (MIP, middle slice)
- 🔧 Multiple output formats (TIFF16, PNG8)
- 🔧 CLAHE and percentile-based contrast adjustment
- 🔧 Works with preprocessing results dictionaries
- 🔧 Configurable auto-resize logic

**Example**:
```python
from colokroll.analysis.segmentation import CellSegmenter

segmenter = CellSegmenter(
    output_dir="results",
    auto_resize=True,
    max_resize_cap=2500
)

result = segmenter.segment_from_file(
    "data.ome.tiff",
    channel_a="Phalloidin",
    channel_b="DAPI",
    channel_weights=(0.8, 0.2),
    projection="mip",  # or "middle"
    output_format="tiff16",  # or "png8"
)

print(f"Mask: {result.mask_path}")
print(f"Found {result.mask_array.max()} cells")
```

## Key Differences

| Feature | Simple Interface | Advanced Interface |
|---------|-----------------|-------------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ One function call | ⭐⭐⭐ Class instantiation + method |
| **Reliability** | ⭐⭐⭐⭐⭐ Proven workflow | ⭐⭐⭐⭐ More options = more ways to fail |
| **Projection** | MIP only | MIP or middle slice |
| **Format** | PNG (works reliably) | TIFF16 or PNG8 |
| **Parameters** | Essential only | Extensive customization |
| **Best for** | Production workflows | Experimentation |

## Configuration

Both interfaces require a Hugging Face token for API authentication:

### Option 1: Environment Variable (Recommended)
```bash
export HUGGINGFACE_TOKEN="hf_..."
```

### Option 2: YAML Config File
Create `colokroll/analysis/segmentation/config.yaml`:
```yaml
huggingface:
  token: "hf_..."
```

⚠️ **Security Note**: Never commit tokens to git! Add to `.gitignore`:
```
**/config.yaml
```

## Troubleshooting

### "RuntimeError: Cellpose Space failed after retries"

**Solution 1**: Try smaller resize values
```python
segment_cells(..., resize_values=(400, 200))
```

**Solution 2**: Increase pause between calls
```python
segment_cells(..., pause_seconds=2.0)
```

**Solution 3**: Check your image quality
- Ensure composite has good contrast
- Try adjusting weights: `phalloidin_weight=0.9, dapi_weight=0.1`
- Preprocess with background subtraction first

### "ImportError: gradio_client"

Install missing dependency:
```bash
pip install gradio_client imageio
```

### Poor Segmentation Results

1. **Check composite quality**: Visualize the weighted composite
2. **Adjust weights**: Try different phalloidin/DAPI ratios
3. **Preprocess first**: Use background subtraction before segmentation
4. **Adjust Cellpose parameters**: Lower `flow_threshold` for more cells

## API Reference

### `segment_cells()`

```python
def segment_cells(
    phalloidin_mip: np.ndarray,
    dapi_mip: np.ndarray,
    output_dir: Union[str, Path],
    filename_stem: str,
    *,
    phalloidin_weight: float = 0.8,
    dapi_weight: float = 0.2,
    resize_values: Tuple[int, ...] = (600, 400),
    max_iter: int = 250,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    pause_seconds: float = 1.0,
    tmp_dir: str = "/tmp",
) -> Tuple[Path, Path, np.ndarray]
```

**Returns**: `(mask_path, outlines_path, mask_array)`

### `segment_from_loader()`

```python
def segment_from_loader(
    image_loader,
    phalloidin_channel: Union[int, str],
    dapi_channel: Union[int, str],
    output_dir: Union[str, Path],
    filename_stem: Optional[str] = None,
    **kwargs
) -> Tuple[Path, Path, np.ndarray]
```

**Returns**: `(mask_path, outlines_path, mask_array)`

## Examples

See `examples/simple_segmentation_example.py` for complete examples including:
- Direct usage with MIPs
- Convenience usage with ImageLoader
- Batch processing multiple images
- Integration with preprocessing

## Implementation Notes

The simple interface (`segment_cells`) implements the exact workflow from the proven notebook:

1. Create weighted composite from MIPs (0.8 × Phalloidin + 0.2 × DAPI)
2. Normalize to [0, 1] range
3. Save as 8-bit PNG
4. Call `/update_button` endpoint
5. Pause 1 second
6. Call `/cellpose_segment` endpoint
7. Retry with smaller resize values if needed

This workflow has been tested extensively and provides reliable results across various image types.

