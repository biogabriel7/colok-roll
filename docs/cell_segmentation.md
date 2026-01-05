# Cell Segmentation

Automated cell segmentation using Cellpose via HuggingFace Spaces API.

---

## Overview

ColokRoll integrates with [Cellpose](https://www.cellpose.org/) for cell segmentation without requiring local GPU installation. The `CellSegmenter` class:

- Uses HuggingFace Spaces API (no local Cellpose installation needed)
- Supports automatic image resizing for large images
- Creates composites from multiple channels
- Returns labeled masks and outlines

---

## Quick Start

```python
import colokroll as cr
from pathlib import Path

# Initialize segmenter
segmenter = cr.CellSegmenter(
    output_dir=Path("./output/cellpose"),
)

# Segment from background-subtracted results
seg = segmenter.segment_from_results(
    results=results,           # Dict from background subtraction
    channel_a="Phalloidin",    # Cell body channel
    channel_b="DAPI",          # Nuclei channel
    channel_weights=(1.0, 0.1),
)

print(f"Detected {seg.mask_array.max()} cells")
print(f"Mask saved to: {seg.mask_path}")
```

---

## CellSegmenter Options

### Initialization

```python
segmenter = cr.CellSegmenter(
    output_dir=Path("./output"),  # Where to save masks/outlines
    
    # Resize options for large images
    auto_resize=True,             # Auto-resize based on image size
    resize_candidates=[800, 600, 400],  # Candidate sizes to try
    max_dimension=1024,           # Max dimension for auto-resize
    
    # Cellpose parameters (passed to API)
    flow_threshold=0.4,           # Flow error threshold
    cellprob_threshold=0.0,       # Cell probability threshold
)
```

### Resizing Behavior

Large images (>1024px) are automatically resized for the API, then masks are resized back:

```python
# Fixed resize candidates (no auto-detection)
segmenter = cr.CellSegmenter(
    auto_resize=False,
    resize_candidates=[600, 400],  # Will try these sizes
)

# Auto-resize based on image dimensions
segmenter = cr.CellSegmenter(
    auto_resize=True,
    max_dimension=800,
)
```

---

## Segmentation Methods

### From Background Subtraction Results

Use results directly from `BackgroundSubtractor`:

```python
# Background subtraction first
results = {}
for i, ch in enumerate(channel_names):
    corrected, meta = bg_subtractor.subtract_background(
        image=filtered_image[:, :, :, i],
        channel_name=ch,
    )
    results[ch] = (corrected, meta)

# Segment from results
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",       # Primary channel (cell body)
    channel_b="DAPI",             # Secondary channel (nuclei)
    channel_weights=(0.8, 0.2),   # Weight for composite
    projection="mip",             # "mip" or "sme"
    output_format="png8",         # Output format for visualization
    save_basename="sample_001",   # Base name for output files
)
```

### From Image Array

Segment directly from a numpy array:

```python
seg = segmenter.segment_from_image_array(
    image=filtered_image,         # (Z, Y, X, C) array
    channel_indices=(2, 0),       # Indices: (cell body, nuclei)
    channel_weights=(0.8, 0.2),
    projection="mip",
    save_basename="sample_001",
)
```

### From File Path

Segment from an image file:

```python
seg = segmenter.segment_from_file(
    image_path=Path("path/to/image.ome.tiff"),
    channel_indices=(2, 0),
    channel_weights=(0.8, 0.2),
)
```

---

## Creating Composites

The segmenter creates a weighted composite from two channels:

```python
# Composite formula: w1 * ch_a + w2 * ch_b
# Example with weights (0.8, 0.2):
# composite = 0.8 * Phalloidin + 0.2 * DAPI
```

### Weight Selection Tips

| Use Case | Recommended Weights |
|----------|-------------------|
| Strong cell body signal | (1.0, 0.0) |
| Add nuclei hint | (0.8, 0.2) or (0.9, 0.1) |
| Balance both | (0.5, 0.5) |

```python
# Strongly favor Phalloidin (avoids nuclei dominating)
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
    channel_weights=(1.0, 0.10),
)
```

---

## Output

### SegmentationResult

```python
seg = segmenter.segment_from_results(...)

# Paths to saved files
seg.mask_path      # Path to mask TIFF (uint16 labels)
seg.outlines_path  # Path to outlines PNG

# Numpy arrays
seg.mask_array     # (Y, X) array with cell labels (0=background)
seg.outlines       # (Y, X) array with outline pixels

# Metadata
seg.n_cells        # Number of detected cells
seg.composite_shape  # Shape of input composite
```

### Output Files

```
output/cellpose/
├── sample_001_masks.tif       # Labeled mask (uint16)
├── sample_001_outlines.png    # Outline visualization
└── sample_001_composite.png   # Input composite (optional)
```

---

## Projections

### Maximum Intensity Projection (MIP)

Default projection that takes the maximum value along Z:

```python
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
    projection="mip",
)
```

### Surface Manifold Extraction (SME)

For 2.5D objects, SME preserves more signal:

```python
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
    projection="sme",
    sme_reference_channel="Phalloidin",  # Channel for manifold
)
```

---

## Integration with Pipeline

### Complete Workflow

```python
import colokroll as cr
from pathlib import Path

# 1. Load and preprocess
loader = cr.ImageLoader()
image = loader.load_image("path/to/image.ome.tiff")
loader.rename_channels(['DAPI', 'ALIX', 'Phalloidin', 'LAMP1'])

# 2. Z-slice selection
result = cr.select_z_slices(image, strategy="closest", keep_top=14)
filtered_image = image[result.indices_keep]

# 3. Background subtraction
bg_subtractor = cr.BackgroundSubtractor()
results = {}
for i, ch in enumerate(loader.get_channel_names()):
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
    channel_weights=(1.0, 0.1),
    save_basename=Path(image_path).stem,
)

# 5. Use mask for colocalization
mask_path = seg.mask_path
```

---

## Troubleshooting

### API Connection Issues

The segmenter uses HuggingFace Spaces, which requires internet access:

```python
# Check if API is accessible
import gradio_client
client = gradio_client.Client("mouseland/cellpose")
```

### Large Images

For very large images (>2048px), use aggressive resizing:

```python
segmenter = cr.CellSegmenter(
    output_dir=Path("./output"),
    auto_resize=False,
    resize_candidates=[400, 300],  # Smaller sizes
)
```

### Poor Segmentation

Adjust Cellpose parameters:

```python
segmenter = cr.CellSegmenter(
    flow_threshold=0.6,        # Increase for stricter boundaries
    cellprob_threshold=0.5,    # Increase for fewer, more confident cells
)
```

Or try different channel weights:

```python
# Try cell body only
seg = segmenter.segment_from_results(
    results=results,
    channel_a="Phalloidin",
    channel_b="DAPI",
    channel_weights=(1.0, 0.0),  # Ignore nuclei
)
```

